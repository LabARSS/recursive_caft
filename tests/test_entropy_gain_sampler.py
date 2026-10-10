"""Sampling semantics: entropy-gain scoring and top-k selection (pure pandas)."""

from math import nan

import pandas as pd
import pytest
from datasets import Dataset

from core.dataset_samplers.base_sampler import BaseDatasetSamplerConfig
from core.dataset_samplers.entropy_gain_sampler import EntropyGainSampler


def _df(entropy_values, teacher=0.1):
    return pd.DataFrame(
        {
            "id": [f"r{i}" for i in range(len(entropy_values))],
            "entropy_value": list(entropy_values),
            "teacher_entropy": [teacher] * len(entropy_values),
        }
    )


def test_score_is_clamped_gain():
    sampler = EntropyGainSampler(BaseDatasetSamplerConfig(top_k=1))
    assert sampler._score_row({"entropy_value": 0.9, "teacher_entropy": 0.2}) == pytest.approx(0.7)
    # Negative gain clamps to 0.
    assert sampler._score_row({"entropy_value": 0.1, "teacher_entropy": 0.5}) == 0.0


def test_create_sample_selects_top_k_by_gain():
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "entropy_value": [1.0, 0.5, 0.9, 0.2],
            "teacher_entropy": [0.1, 0.1, 0.1, 0.1],
        }
    )  # gains: a=0.9, b=0.4, c=0.8, d=0.1
    sampler = EntropyGainSampler(BaseDatasetSamplerConfig(top_k=2))

    result = sampler.create_sample(Dataset.from_pandas(df))

    assert len(result) == 2
    assert set(result["id"]) == {"a", "c"}


def test_drop_non_positive_drops_zero_and_negative_gain():
    # drop_non_positive is on by default. r0 gain=+0.4 (kept); r1 gain=0 (entropy==teacher, dropped);
    # r2 gain<0 -> clamped to 0 (dropped).
    df = _df([0.5, 0.1, 0.05], teacher=0.1)
    sampler = EntropyGainSampler(BaseDatasetSamplerConfig(top_k=10))

    result = sampler.create_sample(Dataset.from_pandas(df))

    assert set(result["id"]) == {"r0"}


def test_drop_non_positive_drops_nan_score():
    # An unmeasured row (NaN entropy) scores NaN; NaN > 0 is False, so it is dropped.
    df = _df([0.5, nan], teacher=0.1)
    sampler = EntropyGainSampler(BaseDatasetSamplerConfig(top_k=10))

    result = sampler.create_sample(Dataset.from_pandas(df))

    assert set(result["id"]) == {"r0"}


def test_returns_fewer_than_top_k_when_few_positive():
    # Only 2 of 5 rows have positive gain; selection is the smaller set, not padded to top_k.
    df = _df([0.5, 0.4, 0.1, 0.05, 0.1], teacher=0.1)  # positives: r0, r1
    sampler = EntropyGainSampler(BaseDatasetSamplerConfig(top_k=4))

    result = sampler.create_sample(Dataset.from_pandas(df))

    assert len(result) == 2 < 4
    assert set(result["id"]) == {"r0", "r1"}


def test_raises_when_all_non_positive():
    df = _df([0.1, 0.05, 0.1], teacher=0.1)  # gains: 0, <0->0, 0
    sampler = EntropyGainSampler(BaseDatasetSamplerConfig(top_k=4))

    with pytest.raises(RuntimeError, match="no training samples remain"):
        sampler.create_sample(Dataset.from_pandas(df))


def test_flag_off_preserves_padding_behaviour():
    # With the drop disabled, zero/negative-gain rows are kept to fill top_k (the original behaviour).
    df = _df([0.5, 0.1, 0.05], teacher=0.1)  # one positive, two non-positive
    sampler = EntropyGainSampler(BaseDatasetSamplerConfig(top_k=3, drop_non_positive=False))

    result = sampler.create_sample(Dataset.from_pandas(df))

    assert len(result) == 3
    assert set(result["id"]) == {"r0", "r1", "r2"}


@pytest.mark.parametrize(
    "entropy_values",
    [
        [0.9, 0.5, 0.3],  # all positive
        [0.5, 0.1, 0.05],  # mixed (one positive)
        [0.5, nan, 0.3],  # with NaN
    ],
)
def test_count_selected_matches_create_sample(entropy_values):
    df = _df(entropy_values, teacher=0.1)
    sampler = EntropyGainSampler(BaseDatasetSamplerConfig(top_k=2))

    assert sampler.count_selected(df) == len(sampler.create_sample(Dataset.from_pandas(df)))


def test_count_selected_zero_when_all_non_positive_does_not_raise():
    df = _df([0.1, 0.05], teacher=0.1)
    sampler = EntropyGainSampler(BaseDatasetSamplerConfig(top_k=4))

    assert sampler.count_selected(df) == 0


def test_select_does_not_mutate_input_df():
    df = _df([0.9, 0.5, 0.3], teacher=0.1)
    cols_before = list(df.columns)
    sampler = EntropyGainSampler(BaseDatasetSamplerConfig(top_k=2))

    sampler.count_selected(df)

    assert "score" not in df.columns
    assert list(df.columns) == cols_before
