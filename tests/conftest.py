"""Shared fixtures/helpers for the CPU-only resampling-pipeline tests.

These tests lock in the data-flow contracts of the entropy_gain experiment
(`src/experiments/distillation_by_metrics/mmlu/entropy_gain/qwen_3b.py`) without
any GPU, base model, or flash-attn — only pandas + a tokenizer. The tokenizer is
the same one the experiment uses (`Qwen/Qwen2.5-3B-Instruct`, tokenizer files only,
already in the HF cache), put through `setup_thinking_tokens` exactly like the script.
"""

import pandas as pd
import pytest
from transformers import AutoTokenizer

from core.training.thinking_tokens import setup_thinking_tokens

TOKENIZER_ID = "Qwen/Qwen2.5-3B-Instruct"


@pytest.fixture(scope="session")
def thinking_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    setup_thinking_tokens(tokenizer)
    return tokenizer


def _resampling_row(i: int = 0, **overrides) -> dict:
    """A row shaped like one line of the per-epoch complexity-estimation output:
    the original MMLU/distill columns + entropy_value (student) + teacher_entropy."""
    row = {
        "question_id": f"q{i}",
        "base_cluster": "world geography",
        "question": "What is the capital of France?",
        "options": "['Paris', 'London', 'Berlin', 'Rome']",
        "answer": "a",
        "distill_reasoning": "The capital of France is Paris.",
        "distill_answer": "a",
        "entropy_value": 0.8,
        "teacher_entropy": 0.3,
    }
    row.update(overrides)
    return row


@pytest.fixture
def make_resampling_row():
    return lambda **overrides: _resampling_row(0, **overrides)


@pytest.fixture
def make_resampling_df():
    def _make(n: int) -> pd.DataFrame:
        # Descending entropy so top-k-by-gain selection is deterministic.
        rows = [_resampling_row(i, entropy_value=0.9 - 0.1 * i, teacher_entropy=0.2) for i in range(n)]
        return pd.DataFrame(rows)

    return _make
