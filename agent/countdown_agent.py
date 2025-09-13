import gc
import time

import numpy as np
from tqdm import trange
from typing import List, Dict, Any, Tuple, Union

from deepspeed import DeepSpeedEngine
from transformers import AutoTokenizer, PreTrainedModel

from utils.preprocessing import prepare_inputs
from utils.reward_functions import compute_reward
from agent.countdown_config import CountdownAgentConfig

from datasets import Dataset

import torch

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams

class CountdownAgent():
    def __init__(self, config: CountdownAgentConfig, train_ds: Dataset):
        self.model_name = config.model_name
        self.num_gen_per_sample = config.num_gen_per_sample
        self.episodes_per_iteration = config.episodes_per_iteration
        self.n_epochs = config.n_epochs
        self.per_device_batch_size = config.per_device_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_chat_name) # why do we want model chat name
        self.eos_token_id = AutoTokenizer.from_pretrained(config.model_name).eos_token_id
        self.eos_token = self.tokenizer.convert_ids_to_tokens(self.eos_token_id)

        self.ignore_idx = -100
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.top_k = config.top_k
        self.max_response_tokens = config.max_response_tokens

        self.metrics = {}

        # DeepSpeed config for the policy model
        self.deepspeed_config = {
            "bf16": {"enabled": True},
            "zero_optimization": {"stage": 2, "overlap_comm": False},
            "train_batch_size": config.episodes_per_epoch,
            "train_micro_batch_size_per_gpu": config.per_device_batch_size,
            "gradient_accumulation_steps": config.episodes_per_epoch // config.per_device_batch_size,
            "gradient_clipping": 1.0,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.config.learning_rate,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                    "torch_adam": True,
                },
            },
        }
        # DeepSpeed config for the reference model
        self.ref_deepspeed_config = {
            "bf16": {"enabled": True},
            # Note that we don't train the reference model
            # These are just for compatibility with DeepSpeed.
            "train_batch_size": config.episodes_per_epoch,
            "train_micro_batch_size_per_gpu": config.per_device_batch_size,
            "gradient_accumulation_steps": config.episodes_per_epoch // config.per_device_batch_size,
        }

        # Initialize main and reference models
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map=0,
        )
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map=0,
        )
        self.policy_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


        # Initialize DeepSpeed engines
        self.policy_model, *_ = deepspeed.initialize(
            model=self.policy_model,
            config=deepspeed_config,
            model_parameters=self.policy_model.parameters(),
        )
        self.reference_model, *_ = deepspeed.initialize(
            model=self.reference_model,
            config=ref_deepspeed_config,
        )

        self.reference_model.module.cpu()

        self.inference_engine = LLM(
            model=MODEL_NAME,
            skip_tokenizer_init=False,
            gpu_memory_utilization=0.2,
            enable_prefix_caching=True,
            swap_space=1,
            scheduling_policy="fcfs",
            dtype=torch.bfloat16,
            max_model_len=2048,
            enable_sleep_mode=True,
        )

        self.train_ds = train_ds

    def compute_log_probs(
        self,
        model: Union[DeepSpeedEngine, PreTrainedModel],
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns per-token log-probabilities for the *observed* tokens,
        masked to zero where labels are invalid per labels_msk (and/or ignore_idx).

        Expects:
        inputs["input_ids"] : [B, T]
        inputs["attn_msk"]  : [B, T]           (optional but recommended)
        inputs["labels"]    : [B, T]
        inputs["labels_msk"]: [B, T]           (bool or 0/1)
        """

        # 1) Forward pass
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attn_msk", None),
            return_dict=True,
            use_cache=False,
        )

        # 2) Temperature scale and shift for CLM
        logits = outputs.logits / self.temperature            # [B, T, V]
        shift_logits = logits[:, :-1, :]                      # [B, T-1, V]
        shift_labels = inputs["labels"][:, 1:]                # [B, T-1]
        shift_labels_msk = inputs["labels_msk"][:, 1:]        # [B, T-1] (bool/0-1)

        # 3) Compute per-token negative log-likelihood, then negate to get log-prob
        #    Use ignore_index so we don’t need to mutate labels.
        loss_flat = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),   # [(B*(T-1)), V]
            shift_labels.reshape(-1),                          # [(B*(T-1))]
            reduction="none",
            ignore_index=self.ignore_idx,
        )
        logp = -loss_flat.reshape(shift_labels.shape)          # [B, T-1]

        # 4) Mask out invalid positions to exactly 0.0
        #    (Note: they’ll be <= 0 anyway; we zero them for downstream clarity.)
        if shift_labels_msk.dtype != torch.bool:
            shift_labels_msk = shift_labels_msk.bool()

        # Also zero where label == ignore_idx just in case labels_msk missed it
        valid = shift_labels_msk & (shift_labels != self.ignore_idx)
        logp = logp * valid.to(logp.dtype)

        return logp

    def process_rollout(
        self,
        samples: List[Dict[str, Any]],
        generations: List[List[int]],
        finish_reasons: List[str],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Process data collected during rollout to use it for training.
        We calculate rewards by grouping generations,
        calculating their rewards, and normalize them to get eadvantages.
        Args:
            samples: List of input samples containing:
                -input_ids: List[int]
                -nums: List[int]
                -target: int
        Returns:
            Tuple with:
            1. Dictionary with processed data for training:
                - query_token_ids: List[Dict[str, int]], input token IDs repeated for each generations
                - response_token_ids: List[List[int]], response token IDs with EOS tokens added
                - adv: List[List[float]], advantage values repeated for each token
            2. Dictionary with generation statistics:
                - response_lengths: List[int], length of generated responses
                - rewards: List[float], raw reward values
                - non_stop_rate: List[bool], whether each generation ended
                - reward_metrics: reward component metrics
        Example:
            >>> samples = [{"input_ids": [1,2,3], "nums": [1,2,3], "target": 6}]
            >>> generations = [[4,5, EOS_TOKEN_ID], [6,7], [8,9, EOS_TOKEN_ID]]  # 3 generations per sample
            >>> finish_reasons = ["stop", "length", "stop"]
            >>> episodes, stats = create_training_episodes(samples, generations, finish_reasons)
            >>> episodes
            {
                'query_token_ids': [[1,2,3], [1,2,3], [1,2,3]],
                'response_token_ids': [[4,5,EOS_TOKEN_ID], [6,7], [8,9,EOS_TOKEN_ID]],
                'adv': [[0.5,0.5,0.5], [-1.0,-1.0], [0.5,0.5,0.5]]
            }
        """

        assert len(generations) == len(finish_reasons)
        assert len(generations) == len(samples) * self.num_gen_per_sample

        query_token_ids: List[List[int]] = []
        response_token_ids: List[List[int]] = []
        adv: List[List[float]] = []

        stats: Dict[str, Any] = {
            "response_lengths": [],
            "rewards": [],
            "non_stop_rate": [],
        }

        for s_idx, sample in enumerate(samples):
            start = s_idx * self.num_gen_per_sample
            end = start + self.num_gen_per_sample

            s_response_token_ids = generations[start:end]

            # Decode all responses for this sample at once
            s_responses = self.tokenizer.batch_decode(s_response_token_ids, skip_special_tokens=False)

            # Compute rewards & metrics
            rewards_and_metrics = [compute_reward(resp, sample) for resp in s_responses]
            rewards, reward_metrics = zip(*rewards_and_metrics)  # tuples length self.num_gen_per_sample

            rewards_np = np.asarray(rewards, dtype=np.float32)
            std = rewards_np.std()
            response_adv = (rewards_np - rewards_np.mean()) / (std + 1e-4)

            lengths = [len(ids) for ids in s_response_token_ids]
            s_adv = [[float(a)] * L for a, L in zip(response_adv, lengths)]

            # Accumulate outputs
            query_token_ids.extend([sample["input_ids"]] * self.num_gen_per_sample)
            response_token_ids.extend(s_response_token_ids)
            adv.extend(s_adv)

            stats["rewards"].extend(rewards_np.tolist())
            stats["response_lengths"].extend(lengths)

            for rm in reward_metrics:
                for k, v in rm.items():
                    key = f"reward_metrics/{k}"
                    if key in stats:
                        stats[key].append(v)
                    else:
                        stats[key] = [v]

        stats["non_stop_rate"].extend([fr != "stop" for fr in finish_reasons])
        
        episodes = {
            "query_token_ids": query_token_ids,
            "response_token_ids": response_token_ids,
            "adv": adv,
        }
        return episodes, stats
    
    def compute_loss(
        batch: Dict[str, torch.Tensor],
        total_response_len: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute policy gradient loss with KL penalty between policy and reference model.

        1. compute log probs for policy and reference models
        2. calculate KL divergence between policy and reference model
        3. compute policy gradient loss
        4. combine losses

        Args:
            policy_model: model being trained
            reference_model: reference model for KL penalty
            batch: Dict containing:
                - input_ids: tensor of shape [batch_size, seq_len]
                - attn_msk: tensor of shape [batch_size, seq_len]
                - labels: tensor of shape [batch_size, seq_len]
                - labels_msk: mask indicating valid labels
                - adv: tensor of shape [batch_size, seq_len]

        Returns:
            Tuple containing:
                - loss: combined policy gradient and kl penalty loss
                - metrics: dict with loss components
                    - policy_loss: Pure policy gradient loss
                    - kl_penalty: KL divergence penalty
                    - entropy: policy entropy
        """
        input_ids = batch["input_ids"]
        attn_msk = batch["attn_msk"]
        labels = batch["labels"]
        adv = batch["adv"]

        model_inputs = {
            "input_ids": input_ids,
            "atn_msk": attn_msk,
            "labels": labels,
        }

        # [batch_size, seq_len-1]
        logs compute_log_probs(
            policy_model, model_inputs, self.temperature
        )

        with torch.no_grad():
            # [batch_size, seq_len-1]
            ref_logs = compute_log_probs(
                reference_model, model_inputs, self.temperature
            )


    def rollout(
            self,
    ) -> List[Dict[str, Any]]:
        """
        Generate rollouts by sampling from the model.

        Args:
            None

        Returns:
            List of generated samples
        """
        num_samples = self.episodes_per_epoch // self.num_gen_per_sample
        indices = np.random.randint(
            0, len(self.train_ds), size=num_samples, replace=False
        )
        samples = self.train_ds.select(indices)

        outputs = inference_engine.generate(
            prompt_token_ids=samples["input_ids"],
            sampling_params=SamplingParams(
                n=self.num_gen_per_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_tokens=self.max_response_tokens,
                detokenize=False,
                stop_token_ids=[self.eos_token_id],
            )
        )
        generations = [list(g.token_ids) for out in outputs for g in out.outputs]
        finish_reasons = [list(g.finish_reasons) for out in outputs for g in out.outputs]
        self.inference_engine.sleep(1)

        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        return samples, generations, finish_reasons
    
        
    def training_epoch():

        # rollout
        samples, generations, finish_reasons = self.rollout()

        episodes, stats = self.process_rollout(
            samples,
            generations,
            finish_reasons
        )

        for k, v in episodes_stats.items():
            metrics.setdefault(k, []).extend(v)

        # prepare inputs
        inputs = prepare_inputs(
            query_token_ids=episodes["query_token_ids"],
            response_token_ids=episodes["response_token_ids"],
            adv=episodes["adv"],
            device=self.device,
            ignore_idx=self.ignore_idx
        )

        self.policy_model.train()
        self.ref_model.train()
        self.ref_model.eval()

        total_response_len = inputs["labels_msk"].sum().item()

        for i in trange(0, self.episodes_per_epoch, self.per_device_batch_size):
            batch = {
                k: v[i : i + self.per_device_batch_size]
                for k, v in inputs.items()
            }
            loss, loss_metrics = self.compute_loss(
                batch,
                total_response_len,
            )

            # Track metrics
            metrics.setdefault("loss", []).append(loss.item())
            grad_norm = policy_model.get_global_grad_norm()
            if grad_norm is not None:
                grad_norm = grad_norm.item()
            metrics.setdefault("grad_norm", []).append(grad_norm)
            for k, v in loss_metrics.items():
                metrics.setdefault(k, []).append(v.item() if isinstance(v, torch.Tensor) else v)

            # Backpropagation and optimization step
            policy_model.backward(loss, scale_wrt_gas=False)
            
            # Free memory
            del loss, loss_metrics
            if policy_model.is_gradient_accumulation_boundary():
                reference_model.module.cpu()

            policy_model.step()

        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        inference_engine.wake_up()
        load_model_into_vllm(self.policy_model, self.inference_engine)

        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)


    def train(self):
        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch+1}/{self.n_epochs}")
            self.training_epoch()
            # Log epoch metrics
            epoch_metrics = {k: np.mean(v) for k, v in self.metrics.items()}
            print(f"Epoch {epoch+1} metrics: {epoch_metrics}")
            self.metrics = {}

if __name__ == "__main__":
    agent_config = CountdownAgentConfig()
    agent_config.num_gen_per_sample = 3
    agent = CountdownAgent(agent_config)

    samples = [
        {"input_ids": [1,1,1], "nums": [0,0,0], "target": 0},
        {"input_ids": [2,2,2], "nums": [0,0,0], "target": 0}
    ]

    # 3 generations per sample
    generations = [
        [4, 5, agent.eos_token_id],
        [6, 7],
        [8, 9, agent.eos_token_id],
        [10, 5, 1, agent.eos_token_id],
        [6, 7, 1],
        [8, 9, agent.eos_token_id]
    ]

    finish_reasons = ["stop", "length", "stop", "stop", "stop", "length"]

    episodes, stats = agent.process_rollout(samples, generations, finish_reasons)

    print(samples)
    print("==========")
    print(generations)
    print("==========")
    print(episodes)
    print("==========")
    print(stats)