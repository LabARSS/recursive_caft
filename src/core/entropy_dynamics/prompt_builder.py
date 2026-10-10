"""
Two modes:
  FORCED (A):  <think>r_1...r_{kn}</think>  The answer is: student outputs 1 token, we measure its entropy.

  CONTINUATION (B):  <think>r_1...r_{kn} student continues until [[answer]], we measure average entropy of last 5 tokens before ]].
"""

from __future__ import annotations

import string
from dataclasses import dataclass

from transformers import PreTrainedTokenizer

from core.entropy_dynamics.config import InferenceMode
from core.entropy_dynamics.reasoning_loader import TeacherReasoning
from core.prompts.thinking_markers import THINKING_START, THINKING_END

OPTION_IDS = list(string.ascii_lowercase)


@dataclass
class PrefixedPrompt:
    """One prompt for a specific question at prefix step k."""
    question_id: str
    k: int                    # prefix step index 
    num_reasoning_tokens: int  # how many teacher tokens included
    total_reasoning_tokens: int  # total available
    input_ids: list[int]
    gold_answer: str
    mode: InferenceMode


def build_prefixed_prompts(
    sample: TeacherReasoning,
    tokenizer: PreTrainedTokenizer,
    window_size: int,
    mode: InferenceMode,
    dataset_type: str = "mmlu",
) -> list[PrefixedPrompt]:
    """Generate prompts for k=0,1,...,K for one question.
    """
    T = len(sample.thinking_token_ids)
    K = T // window_size 

    sys_prompt = _system_prompt(dataset_type, sample)
    user_prompt = _user_prompt(dataset_type, sample)

    prompts: list[PrefixedPrompt] = []

    # k=0: no reasoning at all
    messages_k0 = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if mode == InferenceMode.FORCED:
        assistant_prefix = f"{THINKING_START}{THINKING_END}\nThe answer is:"
        input_ids = _tokenize_with_assistant_prefix(tokenizer, messages_k0, assistant_prefix)
    else:
        # Continuation: just the question, no think tag at k=0
        input_ids = tokenizer.apply_chat_template(
            messages_k0, tokenize=True, add_generation_prompt=True
        )

    prompts.append(PrefixedPrompt(
        question_id=sample.question_id,
        k=0,
        num_reasoning_tokens=0,
        total_reasoning_tokens=T,
        input_ids=input_ids,
        gold_answer=sample.gold_answer,
        mode=mode,
    ))

    # k=1..K: incremental reasoning prefixes
    for k in range(1, K + 1):
        n_tokens = k * window_size
        reasoning_prefix_ids = sample.thinking_token_ids[:n_tokens]
        reasoning_text = tokenizer.decode(reasoning_prefix_ids, skip_special_tokens=True)

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if mode == InferenceMode.FORCED:
            assistant_prefix = f"{THINKING_START}{reasoning_text}{THINKING_END}\nThe answer is:"
            input_ids = _tokenize_with_assistant_prefix(tokenizer, messages, assistant_prefix)
        else:
            # Open think tag — student continues reasoning
            assistant_prefix = f"{THINKING_START}{reasoning_text}"
            input_ids = _tokenize_with_assistant_prefix(tokenizer, messages, assistant_prefix)

        prompts.append(PrefixedPrompt(
            question_id=sample.question_id,
            k=k,
            num_reasoning_tokens=n_tokens,
            total_reasoning_tokens=T,
            input_ids=input_ids,
            gold_answer=sample.gold_answer,
            mode=mode,
        ))

    return prompts


def _tokenize_with_assistant_prefix(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict],
    assistant_prefix: str,
) -> list[int]:
    """Tokenize messages + beginning of assistant response.
    """
    base_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    
    # FIX: Flatten BatchEncoding if necessary
    if not isinstance(base_ids, list):
        base_ids = base_ids.input_ids[0] if isinstance(base_ids.input_ids[0], list) else list(base_ids.input_ids)

    prefix_ids = tokenizer.encode(assistant_prefix, add_special_tokens=False)
    return base_ids + prefix_ids


def _system_prompt(dataset_type: str, sample: TeacherReasoning) -> str:
    if dataset_type == "mmlu":
        return (
            "The following are multiple choice questions. "
            "Choose a correct option letter. "
            "Answer with a single symbol. Do not print anything else."
        )
    elif dataset_type == "gpqa":
        return (
            "The following are multiple choice questions. "
            "Choose a correct option letter. "
            "Answer with a single symbol. Do not print anything else."
        )
    elif dataset_type == "gsm8k":
        return (
            "The following are grade school math word problems. "
            "Please, return your answer as a single number "
            "(without extra/special symbols) and nothing else."
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")


def _user_prompt(dataset_type: str, sample: TeacherReasoning) -> str:
    """Format question + options."""
    question = sample.question.strip()

    if dataset_type in ("mmlu", "gpqa"):
        options_str = "\n".join(
            f"{oid}. {text}".strip()
            for oid, text in zip(OPTION_IDS, sample.options)
        )
        return f"Question: {question}\nOptions:\n{options_str}\n"

    elif dataset_type == "gsm8k":
        return question

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
