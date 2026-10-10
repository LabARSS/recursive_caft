"""The estimator is the only per-epoch hook, so EntropyGainProportionalSampler's draw only exists if
SingleTokenEntropyWithRandomEstimator actually lands a fresh `random_value` column next to
`entropy_value` in the per-epoch parquet the sampler reads back.
"""

from types import SimpleNamespace

import pandas as pd
import torch

from core.complexity_estimation.complexity_estimation_runner import (
    ComplexityEstimationRunner,
    ComplexityEstimationRunnerConfig,
    ModelGenerateConfig,
)
from core.complexity_estimation.entropy.single_token_entropy_with_random_estimator import (
    SingleTokenEntropyWithRandomEstimator,
)
from core.dataset_samplers.base_sampler import BaseDatasetSamplerConfig
from core.dataset_samplers.entropy_gain_proportional_sampler import EntropyGainProportionalSampler
from core.datasets.mmlu.mmlu_single_token_response_dataset import MMLUSingleTokenResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter

ROWS = 8


class _FakeModel:
    """Appends the token(s) for "a" and returns flat (defined-entropy) scores."""

    def __init__(self, tokenizer):
        self.device = "cpu"
        self._answer_tokens = tokenizer.encode("a", add_special_tokens=False)
        self._vocab = len(tokenizer)

    def generate(self, input_ids, attention_mask, **kwargs):
        appended = torch.tensor([self._answer_tokens], dtype=input_ids.dtype, device=input_ids.device)
        sequences = torch.cat([input_ids, appended], dim=1)
        return SimpleNamespace(sequences=sequences, scores=(torch.zeros(1, self._vocab),))


def test_estimator_writes_entropy_and_random_value(thinking_tokenizer, tmp_path):
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
        complexity_estimator=SingleTokenEntropyWithRandomEstimator(),
    )

    base = {
        "base_cluster": "geo",
        "question": "What is the capital of France?",
        "options": "['Paris','London','Berlin','Rome']",
        "answer": "a",
        "teacher_entropy": 0.3,
        "estimation_phase_answer": None,
        "estimation_phase_answer_correctness": None,
    }
    pd.DataFrame([{**base, "question_id": f"q{i}"} for i in range(ROWS)]).to_parquet(runner.tmp_path())

    runner.estimate(dataset_adapter=adapter, model=_FakeModel(thinking_tokenizer))

    res = pd.read_parquet(out_path)
    assert {"entropy_value", "random_value"} <= set(res.columns)
    assert res["entropy_value"].notna().all()
    assert res["random_value"].between(0.0, 1.0, inclusive="left").all()
    # A per-row draw, not one value broadcast over the epoch.
    assert res["random_value"].nunique() == ROWS

    # The parquet the sampler reads back is directly scorable: flat logits give max entropy (1.0
    # normalized), so every row has a positive gain over the 0.3 teacher and stays eligible.
    sampler = EntropyGainProportionalSampler(BaseDatasetSamplerConfig(top_k=4))
    assert sampler.count_selected(res) == 4
