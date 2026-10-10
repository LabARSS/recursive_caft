"""Regression: resuming complexity estimation from a `.tmp` must measure exactly the rows that were
not yet measured before the crash — no more, no less.

Missing answers round-trip back from the `.tmp` parquet as `float('nan')`, not `None`. The runner's
skip-check must therefore be missing-value aware (`pd.notna`); a plain `is not None` treats a NaN
(not-yet-measured) row as done and skips it, so a resumed epoch measures nothing and its entropy ends
up stale/backfilled — "we lose the entropy on recovery".
"""

from types import SimpleNamespace

import pandas as pd
import torch

from core.complexity_estimation.complexity_estimation_runner import (
    ComplexityEstimationRunner,
    ComplexityEstimationRunnerConfig,
    ModelGenerateConfig,
)
from core.complexity_estimation.entropy.single_token_entropy_estimator import SingleTokenEntropyEstimator
from core.datasets.mmlu.mmlu_single_token_response_dataset import MMLUSingleTokenResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter


class _FakeModel:
    """Appends the token(s) for "a" and returns flat (defined-entropy) scores, counting calls so the
    test can assert which rows were actually generated for."""

    def __init__(self, tokenizer):
        self.device = "cpu"
        self.calls = 0
        self._answer_tokens = tokenizer.encode("a", add_special_tokens=False)
        self._vocab = len(tokenizer)

    def generate(self, input_ids, attention_mask, **kwargs):
        self.calls += 1
        appended = torch.tensor([self._answer_tokens], dtype=input_ids.dtype, device=input_ids.device)
        sequences = torch.cat([input_ids, appended], dim=1)
        scores = (torch.zeros(1, self._vocab),)
        return SimpleNamespace(sequences=sequences, scores=scores)


def test_resume_measures_only_unmeasured_rows(thinking_tokenizer, tmp_path):
    adapter = QADatasetAdapter(
        dataset=MMLUSingleTokenResponseDataset(
            config=QADatasetConfig(path="sentinel", dataset_id="mmlu_teacher_entropy"),
            tokenizer=thinking_tokenizer,
        )
    )
    out_path = tmp_path / "mmlu_teacher_entropy.parquet"
    runner = ComplexityEstimationRunner(
        config=ComplexityEstimationRunnerConfig(
            out_path=str(out_path),
            answer_field_name="estimation_phase_answer",
            answer_correctness_field_name="estimation_phase_answer_correctness",
            generate_config=ModelGenerateConfig(max_new_tokens=1),
            save_every=1000,
        ),
        complexity_estimator=SingleTokenEntropyEstimator(),
    )

    # A real runner `.tmp`: source columns + stale tokenization columns + partial estimation columns.
    base = {
        "base_cluster": "geo",
        "question": "What is the capital of France?",
        "options": "['Paris','London','Berlin','Rome']",
        "answer": "a",
        "teacher_entropy": 0.3,
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
        "labels": [],
        "row_id": "STALE",
    }
    measured = {**base, "question_id": "q0", "estimation_phase_answer": "a",
                "estimation_phase_answer_correctness": True, "entropy_value": 0.5}
    unmeasured = {**base, "question_id": "q1", "estimation_phase_answer": None,
                  "estimation_phase_answer_correctness": None, "entropy_value": None}
    pd.DataFrame([measured, unmeasured]).to_parquet(runner.tmp_path())

    fake = _FakeModel(thinking_tokenizer)
    runner.estimate(dataset_adapter=adapter, model=fake)

    # Only the not-yet-measured row is generated; the already-measured row is reused.
    assert fake.calls == 1
    res = pd.read_parquet(out_path).set_index("question_id")
    assert res.loc["q0", "entropy_value"] == 0.5  # preserved, not recomputed
    assert pd.notna(res.loc["q1", "entropy_value"])  # freshly measured, not left NaN / backfilled
    # The runner cleans up the .tmp on successful completion.
    assert not runner.tmp_path().exists()
