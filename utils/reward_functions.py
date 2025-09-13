import re
from collections import Counter
from typing import Any, Dict, List, Tuple

# ===== Config =====
EOS_TOKEN = "</s>"
ALLOWED_CHARS = set("0123456789+-*/(). \t\r\n")
NUM_RE = re.compile(r"\d+")

# Strict: require <think>...</think> then <answer>...</answer>, and forbid nested <think> inside think block.
THINK_ANSWER_STRICT = re.compile(
    r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)</think>\s*<answer>(.*?)</answer>$",
    re.DOTALL
)

# Relaxed: just find <answer>...</answer> anywhere.
ANSWER_RELAXED = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _strip_eos(s: str) -> str:
    return s[:-len(EOS_TOKEN)] if s.endswith(EOS_TOKEN) else s


def _parse_answer_strict(completion: str, *, add_think_tag: bool) -> str | None:
    """
    Strict parser for formatting score.
    If add_think_tag=True, the model is expected to output only the closing </think> part
    (opening <think> is prefilled). In that case, we REJECT any completion that contains
    its own '<think>' opening tag.
    """
    completion = _strip_eos(completion)

    if add_think_tag:
        # Disallow user-provided opening <think> entirely
        if "<think>" in completion:
            return None
        # Prepend a single opening <think> for strict parsing
        completion = "<think>" + completion

    m = THINK_ANSWER_STRICT.match(completion)
    if not m or m.lastindex != 2:
        return None
    return m.group(2).strip()


def _parse_answer_relaxed(completion: str) -> str | None:
    """Relaxed parser for equation score: only extract <answer>...</answer>."""
    completion = _strip_eos(completion)
    m = ANSWER_RELAXED.search(completion)
    return m.group(1).strip() if m else None

def format_reward_func(completion: str, *, add_think_tag: bool = True) -> float:
    try:
        answer = _parse_answer_strict(completion, add_think_tag=add_think_tag)
        if answer is None:
            return 0.0
        return 1.0 if all(c in ALLOWED_CHARS for c in answer) else 0.5
    except Exception:
        return 0.0

def equation_reward_func(completion: str, nums: List[int], target: int | float) -> float:
    """
    Scores equation correctness only (independent of format).
    """
    try:
        answer = _parse_answer_relaxed(completion)
        if answer is None:
            return 0.0
        if not all(c in ALLOWED_CHARS for c in answer):
            return 0.0

        used_numbers = [int(n) for n in NUM_RE.findall(answer)]
        if Counter(used_numbers) != Counter(nums):
            return 0.0

        result = eval(answer, {"__builtins__": None}, {})
        return 1.0 if abs(float(result) - float(target)) < 1e-5 else 0.0
    except Exception:
        return 0.0

def compute_reward(
    completion: str,
    sample: Dict[str, Any],
    *,
    add_think_tag: bool = True,   # <— expose the flag here
) -> Tuple[float, Dict[str, float]]:
    nums = sample["nums"]
    target = sample["target"]
    try:
        target = float(target)
    except Exception:
        pass

    # <— forward the flag to format_reward_func
    fmt = format_reward_func(completion, add_think_tag=add_think_tag)
    eqn = equation_reward_func(completion, nums=nums, target=target)  # stays independent

    return fmt + eqn, {"format_reward": fmt, "equation_reward": eqn}


if __name__ == "__main__":
    # correct
    completion = "some thinking... </think><answer>1+2+3</answer>"
    sample = {"nums": [1, 2, 3], "target": 6}
    print(completion)
    print(compute_reward(completion, sample))
    # missing newline
    completion = "some thinking... </think>some stuff<answer>1+2+3</answer>"
    sample = {"nums": [1, 2, 3], "target": 6}
    print(completion)
    print(compute_reward(completion, sample))
    # wrong result
    completion = "some thinking... </think>\n<answer>1+2+3</answer>"
    sample = {"nums": [1, 2, 3], "target": 3}
    print(completion)
    print(compute_reward(completion, sample))
    # additional think tag
    completion = "<think>some thinking... </think>\n<answer>1+2+3</answer>"
    sample = {"nums": [1, 2, 3], "target": 6}
    print(completion)
    print(compute_reward(completion, sample))
    # additional think tag allowed
    completion = "<think>some thinking... </think>\n<answer>1+2+3</answer>"
    sample = {"nums": [1, 2, 3], "target": 6}
    print(completion)
    print(compute_reward(completion, sample, add_think_tag=False))
    # think tag inside think tag
    completion = "<think>some thinking... </think></think><answer>1+2+3</answer>"
    sample = {"nums": [1, 2, 3], "target": 1}
    print(completion)
    print(compute_reward(completion, sample, add_think_tag=False))