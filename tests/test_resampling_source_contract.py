"""BUG C regression: the per-epoch resampling source must tokenize into a
reasoning training example.

During resampling the training source is replaced by the per-epoch
complexity-estimation parquet. That parquet must carry the distill columns
(`distill_reasoning`/`distill_answer`) that `MMLUReasoningResponseDataset.assistant_response`
needs, on top of the entropy columns the sampler scores. If it doesn't,
`CausalDatasetAdapter.process_row` raises KeyError mid-training. These tests pin
that contract using the same tokenizer the experiment uses.
"""

import pytest

from core.dataset_samplers.base_sampler import BaseDatasetSamplerConfig
from core.dataset_samplers.entropy_gain_sampler import EntropyGainSampler
from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.mmlu.mmlu_reasoning_response_dataset import MMLUReasoningResponseDataset
from core.datasets.qa_dataset import QADatasetConfig


def _adapter(tokenizer, sampler=None):
    return CausalDatasetAdapter(
        dataset=MMLUReasoningResponseDataset(
            config=QADatasetConfig(path="sentinel-overridden-at-runtime", dataset_id="t"),
            tokenizer=tokenizer,
        ),
        dataset_sampler=sampler,
    )


def test_process_row_builds_masked_reasoning_example(thinking_tokenizer, make_resampling_row):
    adapter = _adapter(thinking_tokenizer)
    row = make_resampling_row(distill_reasoning="The capital of France is Paris.", distill_answer="A")

    result = adapter.process_row(row)

    ids = result.input_ids
    labels = result.labels
    assert isinstance(ids, list) and len(ids) > 0
    assert len(labels) == len(ids)
    # Prompt prefix is masked, assistant span is supervised.
    assert -100 in labels
    assert any(label != -100 for label in labels)
    assert result.row_id == row["question_id"]

    supervised = thinking_tokenizer.decode([label for label in labels if label != -100])
    assert "<think>" in supervised
    assert "</think>" in supervised
    assert "Paris" in supervised  # the reasoning chain is part of the target


def test_missing_distill_columns_raises(thinking_tokenizer, make_resampling_row):
    # An entropy-only source (no distill columns) is exactly the pre-fix failure.
    adapter = _adapter(thinking_tokenizer)
    row = make_resampling_row()
    del row["distill_reasoning"]

    with pytest.raises(KeyError):
        adapter.process_row(row)


def test_process_dataset_path_override_samples_and_tokenizes(thinking_tokenizer, tmp_path, make_resampling_df):
    # Mirror the real per-epoch path: load the override parquet -> sample top_k -> tokenize.
    parquet = tmp_path / "epoch.parquet"
    make_resampling_df(5).to_parquet(parquet)
    adapter = _adapter(thinking_tokenizer, EntropyGainSampler(BaseDatasetSamplerConfig(top_k=2)))

    ds = adapter.process_dataset(path_override=str(parquet))

    assert len(ds) == 2
    assert "input_ids" in ds.column_names
    assert "labels" in ds.column_names
