"""Sampling semantics: entropy-gain scoring with a proportional (rather than top-k) draw."""

import random
from math import isnan, log, nan

import pandas as pd
import pytest
from datasets import Dataset

from core.dataset_samplers.base_sampler import BaseDatasetSamplerConfig
from core.dataset_samplers.entropy_gain_proportional_sampler import EntropyGainProportionalSampler


def _df(entropy_values, teacher=0.1, random_values=None):
    """A pool shaped like one epoch of complexity-estimation output: the student entropy, the static
    teacher entropy, and the per-epoch uniform draw written by SingleTokenEntropyWithRandomEstimator."""
    if random_values is None:
        random_values = [0.5] * len(entropy_values)
    return pd.DataFrame(
        {
            "id": [f"r{i}" for i in range(len(entropy_values))],
            "entropy_value": list(entropy_values),
            "teacher_entropy": [teacher] * len(entropy_values),
            "random_value": list(random_values),
        }
    )


def _randomize(df):
    df["random_value"] = [random.random() for _ in range(len(df))]
    return df


def test_score_is_gain_over_an_exponential_variate():
    sampler = EntropyGainProportionalSampler(BaseDatasetSamplerConfig(top_k=1))

    score = sampler._score_row({"entropy_value": 0.9, "teacher_entropy": 0.2, "random_value": 0.5})

    assert score == pytest.approx(0.7 / -log(0.5))


def test_score_is_monotone_in_gain_at_a_fixed_draw():
    # Same u: the key is a strictly increasing function of the gain, so the perturbation only
    # reorders rows through their draws, never through the gain itself.
    sampler = EntropyGainProportionalSampler(BaseDatasetSamplerConfig(top_k=1))

    low = sampler._score_row({"entropy_value": 0.4, "teacher_entropy": 0.1, "random_value": 0.3})
    high = sampler._score_row({"entropy_value": 0.9, "teacher_entropy": 0.1, "random_value": 0.3})

    assert high > low


def test_score_is_non_positive_for_non_positive_gain():
    sampler = EntropyGainProportionalSampler(BaseDatasetSamplerConfig(top_k=1))

    # Gain of exactly 0 and negative gain both clamp to 0, so no draw can rescue them.
    assert sampler._score_row({"entropy_value": 0.1, "teacher_entropy": 0.1, "random_value": 0.99}) == 0.0
    assert sampler._score_row({"entropy_value": 0.05, "teacher_entropy": 0.5, "random_value": 0.99}) == 0.0


def test_score_is_nan_for_unmeasured_row():
    sampler = EntropyGainProportionalSampler(BaseDatasetSamplerConfig(top_k=1))

    assert isnan(sampler._score_row({"entropy_value": nan, "teacher_entropy": 0.1, "random_value": 0.5}))


def test_missing_random_value_column_raises():
    # The sampler is only meaningful alongside SingleTokenEntropyWithRandomEstimator; fail loudly
    # rather than silently degrading to a deterministic top-k.
    sampler = EntropyGainProportionalSampler(BaseDatasetSamplerConfig(top_k=1))

    with pytest.raises(KeyError):
        sampler._score_row({"entropy_value": 0.9, "teacher_entropy": 0.2})


def test_nan_random_value_still_eligible():
    # reconcile backfills entropy_value for rows that failed measurement but not random_value, so a
    # positive-gain row with a NaN draw must be redrawn, not dropped.
    df = _df([0.5, 0.6], teacher=0.1, random_values=[nan, 0.5])
    sampler = EntropyGainProportionalSampler(BaseDatasetSamplerConfig(top_k=2))

    result = sampler.create_sample(Dataset.from_pandas(df))

    assert set(result["id"]) == {"r0", "r1"}


def test_only_positive_gain_rows_are_drawn():
    # r0 gain=+0.4 (kept); r1 gain=0 (dropped); r2 gain<0 -> clamped to 0 (dropped); r3 NaN (dropped).
    df = _df([0.5, 0.1, 0.05, nan], teacher=0.1)
    sampler = EntropyGainProportionalSampler(BaseDatasetSamplerConfig(top_k=10))

    result = sampler.create_sample(Dataset.from_pandas(df))

    assert set(result["id"]) == {"r0"}


def test_returns_fewer_than_top_k_when_few_positive():
    df = _df([0.5, 0.4, 0.1, 0.05, 0.1], teacher=0.1)  # positives: r0, r1
    sampler = EntropyGainProportionalSampler(BaseDatasetSamplerConfig(top_k=4))

    result = sampler.create_sample(Dataset.from_pandas(df))

    assert len(result) == 2 < 4
    assert set(result["id"]) == {"r0", "r1"}


def test_raises_when_all_non_positive():
    df = _df([0.1, 0.05, 0.1], teacher=0.1)
    sampler = EntropyGainProportionalSampler(BaseDatasetSamplerConfig(top_k=4))

    with pytest.raises(RuntimeError, match="no training samples remain"):
        sampler.create_sample(Dataset.from_pandas(df))


def test_selection_is_ordered_easy_to_hard():
    random.seed(7)
    df = _randomize(_df([0.9, 0.5, 0.3, 0.7], teacher=0.1))
    sampler = EntropyGainProportionalSampler(BaseDatasetSamplerConfig(top_k=3))

    selected = sampler._select(df)

    assert list(selected["score"]) == sorted(selected["score"])


def test_draw_varies_when_the_epoch_draw_changes():
    # The estimator writes a fresh random_value every epoch; the same pool must then yield a
    # different subset, unlike the deterministic top-k which re-picks the same extreme tail.
    random.seed(0)
    df = _df([0.6] * 40, teacher=0.1)  # all gains equal, so only the draw decides
    sampler = EntropyGainProportionalSampler(BaseDatasetSamplerConfig(top_k=5))

    first = set(sampler._select(_randomize(df))["id"])
    second = set(sampler._select(_randomize(df))["id"])

    assert first != second


def test_selection_probability_is_proportional_to_gain():
    random.seed(1234)
    trials = 2000
    df = _df([0.7, 0.4, 0.2], teacher=0.1)  # gains: 0.6, 0.3, 0.1 -> expected 0.6, 0.3, 0.1
    sampler = EntropyGainProportionalSampler(BaseDatasetSamplerConfig(top_k=1))

    counts = {"r0": 0, "r1": 0, "r2": 0}
    for _ in range(trials):
        selected = sampler._select(_randomize(df))
        counts[selected["id"].iloc[0]] += 1

    assert counts["r0"] / trials == pytest.approx(0.6, abs=0.04)
    assert counts["r1"] / trials == pytest.approx(0.3, abs=0.04)
    assert counts["r2"] / trials == pytest.approx(0.1, abs=0.04)


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
    sampler = EntropyGainProportionalSampler(BaseDatasetSamplerConfig(top_k=2))

    assert sampler.count_selected(df) == len(sampler.create_sample(Dataset.from_pandas(df)))


def test_count_selected_zero_when_all_non_positive_does_not_raise():
    df = _df([0.1, 0.05], teacher=0.1)
    sampler = EntropyGainProportionalSampler(BaseDatasetSamplerConfig(top_k=4))

    assert sampler.count_selected(df) == 0


def test_select_does_not_mutate_input_df():
    df = _df([0.9, 0.5, 0.3], teacher=0.1)
    cols_before = list(df.columns)
    sampler = EntropyGainProportionalSampler(BaseDatasetSamplerConfig(top_k=2))

    sampler.count_selected(df)

    assert "score" not in df.columns
    assert list(df.columns) == cols_before
