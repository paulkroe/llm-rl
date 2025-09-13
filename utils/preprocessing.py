from typing import List, Dict

import torch

from transformers import AutoTokenizer

def create_prompt(
    numbers: List[int],
    target: int,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    prompt_template: str,
) -> str:
    prefix = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": prompt_template.format(numbers=numbers, target=target),
        },
        {"role": "assistant", "content": "Let me reason through this step by step.\n<think>"},
    ]
    return tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True)

def prepare_inputs(
    query_token_ids: List[List[int]],
    response_token_ids: List[List[int]],
    adv: List[List[float]],
    device: torch.device,
    pad_id: int = 0,
    ignore_idx: int = -100,
) -> Dict[str, torch.Tensor]:
    """
    Take rollouts episodes and prepare them as training input to the model.
    Args:
        query_token_ids: List of query token ids
        response_token_ids: List of response token ids
        adv: List of advantages
        device: device to move tensors to
    Returns:
        Dict with input_ids, attn_msk, label_msk, and adv

    Exampe:
        >>> query_token_ids = [[1, 2, 3], [4, 5]]
        >>> response_token_ids = [[6, 7], [8]]
        >>> adv = [[0.5, 0.5], [1]]

        >>> inputs = prepare_inputs(query_token_ids, response_token_ids, adv)
        >>> inputs
            {
                'input_ids': tensor([
                    [1, 2, 3, 6, 7],
                    [4, 5, 8, 0, 0]
                ]), 
                'attn_msk': tensor([
                    [True, True, True, True, True],
                    [True, True, True, False, False]
                ]),
                'labels': tensor([
                    [-100, -100, -100,    6,    7],
                    [-100, -100,    8, -100, -100]
                ]),
                'labels_msk': tensor([
                    [False, False, False, True, True],
                    [False, False, True, False, False]
                ]),
                'adv': tensor([
                    [0.0000, 0.0000, 0.0000, 0.5000, 0.5000],
                    [0.0000, 0.0000, 1.0000, 0.0000, 0.0000]
                ])
            }
    """
    assert len(query_token_ids) == len(response_token_ids) == len(adv)
    B = len(query_token_ids)
    # lengths
    q_lens = torch.tensor([len(q) for q in query_token_ids], dtype=torch.long)
    r_lens = torch.tensor([len(r) for r in response_token_ids], dtype=torch.long)
    seq_lens = q_lens + r_lens
    max_len = int(seq_lens.max().item()) if B > 0 else 0

    input_ids  = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn_msk   = torch.zeros((B, max_len), dtype=torch.bool)
    labels     = torch.full((B, max_len), ignore_idx, dtype=torch.long)
    labels_msk = torch.zeros((B, max_len), dtype=torch.bool)
    adv_pad    = torch.zeros((B, max_len), dtype=torch.float)

    # Fill input_ids and labels row-wise
    for i, (q, r, a) in enumerate(zip(query_token_ids, response_token_ids, adv)):
        q_len, r_len = q_lens[i].item(), r_lens[i].item()
        s_len = q_len + r_len

        # input_ids = [q, r, pad...]
        if s_len:
            input_ids[i, :q_len] = torch.as_tensor(q, dtype=torch.long)
            if r_len:
                input_ids[i, q_len:s_len] = torch.as_tensor(r, dtype=torch.long)

        # labels = [-100 * q_len, response, -100 ...]
        if r_len:
            labels[i, q_len:s_len] = torch.as_tensor(r, dtype=torch.long)

        # advantages: [0 * q_len, a, 0 ...]
        if r_len:
            # tolerate float or double lists
            adv_pad[i, q_len:s_len] = torch.as_tensor(a, dtype=torch.float)

    if max_len > 0:
        pos = torch.arange(max_len, dtype=torch.long).unsqueeze(0)  # [1, L]
        attn_msk = pos < seq_lens.unsqueeze(1)                      # [B, L]
        labels_msk = (pos >= q_lens.unsqueeze(1)) & (pos < seq_lens.unsqueeze(1))

    # Move once to target device
    input_ids  = input_ids.to(device, non_blocking=True)
    attn_msk   = attn_msk.to(device, non_blocking=True)
    labels     = labels.to(device, non_blocking=True)
    labels_msk = labels_msk.to(device, non_blocking=True)
    adv_pad    = adv_pad.to(device, non_blocking=True)

    return {
        "input_ids":  input_ids,
        "attn_msk":   attn_msk,            # bool mask
        "labels":     labels,
        "labels_msk": labels_msk,          # bool mask
        "adv":        adv_pad,             # float
    }

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_token_ids = [[1, 2, 3], [4, 5]]
    response_token_ids = [[6, 7], [8]]
    adv = [[0.5, 0.5], [1.0]]

    print(prepare_inputs(query_token_ids, response_token_ids, adv, device))