from math import log, nan

import pandas as pd
import pytest
from datasets import Dataset
from pydantic import ValidationError

from core.dataset_samplers.annealed_proxy_entropy_sampler import (
    AnnealedProxyEntropySampler,
    AnnealedProxyEntropySamplerConfig,
)


def _sampler(top_k: int = 2, seed: int = 42) -> AnnealedProxyEntropySampler:
    return AnnealedProxyEntropySampler(
        AnnealedProxyEntropySamplerConfig(
            top_k=top_k,
            initial_proxy_weight=0.8,
            reference_epoch=100,
            reference_fraction=0.01,
            seed=seed,
        )
    )


def _df(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "question_id": [f"q{i}" for i in range(n)],
            "entropy_value": [0.2 + i / 100 for i in range(n)],
            "teacher_entropy": [0.1] * n,
        }
    )


def test_initial_proxy_weight_is_required():
    with pytest.raises(ValidationError):
        AnnealedProxyEntropySamplerConfig(top_k=2)


@pytest.mark.parametrize(
    ("epoch", "expected"),
    [
        (0, 0.8),
        (25, 0.8 * 0.01**0.25),
        (50, 0.08),
        (75, 0.8 * 0.01**0.75),
        (100, 0.008),
        (150, 0.0008),
        (200, 0.00008),
    ],
)
def test_proxy_weight_schedule(epoch, expected):
    sampler = _sampler()
    sampler.set_epoch(epoch)
    assert sampler.proxy_weight() == pytest.approx(expected)


def test_score_is_positive_gap_times_deterministic_race_key():
    sampler = _sampler()
    row = {"question_id": "q", "entropy_value": 0.5, "teacher_entropy": 0.25}
    u = sampler._uniform_for("q")
    assert sampler._score_row(row) == pytest.approx((0.5 - 0.8 * 0.25) / -log(u))


@pytest.mark.parametrize(
    ("student", "proxy"),
    [(0.08, 0.1), (nan, 0.1), (0.2, nan), (None, 0.1)],
)
def test_nonpositive_or_nonfinite_gap_is_not_eligible(student, proxy):
    score = _sampler()._score_row(
        {"question_id": "q", "entropy_value": student, "teacher_entropy": proxy}
    )
    assert not score > 0


def test_all_nonpositive_raises_instead_of_falling_back_to_uniform():
    df = pd.DataFrame(
        {"question_id": ["a", "b"], "entropy_value": [0.01, 0.02], "teacher_entropy": [0.1, 0.1]}
    )
    with pytest.raises(RuntimeError, match="no training samples remain"):
        _sampler().create_sample(Dataset.from_pandas(df))


def test_selection_is_reproducible_and_changes_with_epoch_or_seed():
    df = Dataset.from_pandas(_df(100))
    first = _sampler(top_k=20, seed=42)
    same = _sampler(top_k=20, seed=42)
    later = _sampler(top_k=20, seed=42)
    different_seed = _sampler(top_k=20, seed=43)
    later.set_epoch(1)

    ids = set(first.create_sample(df)["question_id"])
    assert ids == set(same.create_sample(df)["question_id"])
    assert ids != set(later.create_sample(df)["question_id"])
    assert ids != set(different_seed.create_sample(df)["question_id"])


def test_smaller_selection_is_nested_when_configuration_matches():
    df = Dataset.from_pandas(_df(100))
    large = _sampler(top_k=20)
    small = _sampler(top_k=5)
    large.set_epoch(37)
    small.set_epoch(37)

    assert set(small.create_sample(df)["question_id"]) < set(large.create_sample(df)["question_id"])
