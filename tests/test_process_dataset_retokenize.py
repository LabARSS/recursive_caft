"""Regression: re-processing a parquet that already carries tokenization columns must re-tokenize
cleanly instead of raising a keyword collision.

The complexity runner periodically flushes progress to a `.tmp` parquet that still holds the
`input_ids/attention_mask/labels/row_id` columns (only its *final* dump drops them). When estimation
resumes from that `.tmp`, `BaseDatasetAdapter.process_dataset` re-runs `process_row`, which passes
those as explicit kwargs — colliding with `**row` ("multiple values for keyword argument 'input_ids'")
unless the stale columns are dropped first. This pins the fix that resumes complexity estimation.
"""

import pandas as pd

from core.datasets.mmlu.mmlu_single_token_response_dataset import MMLUSingleTokenResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter


def _adapter(tokenizer):
    return QADatasetAdapter(
        dataset=MMLUSingleTokenResponseDataset(
            config=QADatasetConfig(path="sentinel", dataset_id="mmlu_teacher_entropy"),
            tokenizer=tokenizer,
        )
    )


def test_process_dataset_retokenizes_over_stale_tokenization(thinking_tokenizer, make_resampling_row, tmp_path):
    adapter = _adapter(thinking_tokenizer)

    # Shape a runner `.tmp` row: original MMLU columns + partial estimation columns + the stale
    # tokenization columns the periodic flush leaves behind.
    row = make_resampling_row()
    row.update(
        {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "labels": [],
            "row_id": "STALE",
            "estimation_phase_answer": "a",
            "estimation_phase_answer_correctness": True,
        }
    )
    parquet = tmp_path / "epoch.tmp"
    pd.DataFrame([row]).to_parquet(parquet)

    ds = adapter.process_dataset(path_override=str(parquet), strict=False)

    out = ds[0]
    # Re-tokenized from source, not the stale sentinel values.
    assert out["input_ids"] != [1, 2, 3]
    assert len(out["input_ids"]) == len(out["attention_mask"])
    assert out["row_id"] != "STALE"
    # Partial estimation columns survive, so the runner can skip already-answered rows on resume.
    assert out["estimation_phase_answer"] == "a"
    assert out["entropy_value"] == row["entropy_value"]
