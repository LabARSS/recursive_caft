"""MMLUSingleTokenResponseDataset.verify_assistant_response: an empty parse (the model emitted
a thinking/special token instead of an answer) is a failed measurement signalled by
InvalidAnswerError — not a cryptic IndexError, and not confused with a wrong-but-valid answer."""

import pytest

from core.datasets.mmlu.mmlu_single_token_response_dataset import MMLUSingleTokenResponseDataset
from core.datasets.qa_dataset import InvalidAnswerError, QADatasetConfig


def _dataset(tokenizer):
    return MMLUSingleTokenResponseDataset(
        config=QADatasetConfig(path="sentinel", dataset_id="x"), tokenizer=tokenizer
    )


def test_empty_answer_raises_invalid_answer_error(thinking_tokenizer):
    ds = _dataset(thinking_tokenizer)
    with pytest.raises(InvalidAnswerError):
        ds.verify_assistant_response({"answer": "a"}, "")


def test_whitespace_answer_raises_invalid_answer_error(thinking_tokenizer):
    ds = _dataset(thinking_tokenizer)
    with pytest.raises(InvalidAnswerError):
        ds.verify_assistant_response({"answer": "a"}, "   ")


def test_correct_letter_is_measured(thinking_tokenizer):
    ds = _dataset(thinking_tokenizer)
    parsed, correct = ds.verify_assistant_response({"answer": "A"}, "a")
    assert parsed == "a"
    assert correct is True


def test_wrong_letter_is_measured_not_failed(thinking_tokenizer):
    # A wrong but parseable single-token answer is a valid (incorrect) measurement, not a failure.
    ds = _dataset(thinking_tokenizer)
    parsed, correct = ds.verify_assistant_response({"answer": "a"}, "b")
    assert parsed == "b"
    assert correct is False
