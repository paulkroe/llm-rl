from dataclasses import dataclass

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. You first think about the reasoning process"
     " and then provide the user with the answer."
)


DEFAULT_PROMPT_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}."
    " You can use basic arithmetic operations (+, -, *, /) and each number can only be used once."
    " Show your work in <think> </think> tags."
    " And return the final equation and answer in <answer> </answer> tags, for example <answer>(1 + 2) / (3 * 5)</answer>."
)

@dataclass
class CountdownAgentConfig:
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    
    model_name: str = "Qwen/Qwen2.5-3B"
    model_chat_name: str = "Qwen/Qwen2.5-3B-Instruct"
 
    num_gen_per_sample: str = 4
    # number of training iterations
    num_iterations: int = 1000
    episodes_per_epoch: int = 64
    per_device_batch_size: int = 4
    kl_coefficient: float = 0.001
    learning_rate: float = 1e-6
    max_response_tokens: int = 1024
    temperature: float = 1.0
    # nucleus sampling parameter (1.0 = disabled)
    top_p: float = 1.0
    # top-k sampling parameter (-1.0 = disabled)
    top_k: float = -1.0

