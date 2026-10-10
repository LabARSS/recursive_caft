"""Proportional sampling from an epoch-annealed student/proxy entropy gap."""

from hashlib import blake2b
from math import exp, isfinite, log
from typing import override

from pydantic import model_validator

from core.dataset_samplers.base_sampler import BaseDatasetSampler, BaseDatasetSamplerConfig


class AnnealedProxyEntropySamplerConfig(BaseDatasetSamplerConfig):
    # Required deliberately: experiments must record the initial proxy contribution explicitly.
    initial_proxy_weight: float
    reference_epoch: int = 100
    reference_fraction: float = 0.01
    seed: int = 42

    @model_validator(mode="after")
    def _validate_schedule(self):
        if not isfinite(self.initial_proxy_weight) or self.initial_proxy_weight < 0:
            raise ValueError("initial_proxy_weight must be finite and nonnegative")
        if self.reference_epoch <= 0:
            raise ValueError("reference_epoch must be positive")
        if not isfinite(self.reference_fraction) or not 0 < self.reference_fraction <= 1:
            raise ValueError("reference_fraction must be finite and in (0, 1]")
        return self


class AnnealedProxyEntropySampler(BaseDatasetSampler):
    """Draw without replacement with weight proportional to a positive entropy gap.

    At effective resampling epoch ``t``, the proxy coefficient is

        initial_proxy_weight * reference_fraction ** (t / reference_epoch).

    A stable per-question exponential-race variate makes the draw reproducible and independent
    of complexity-estimation answer parsing. Samplers with different ``top_k`` but identical
    configuration receive the same keys, so the smaller selection is nested in the larger one.
    """

    config: AnnealedProxyEntropySamplerConfig

    def __init__(self, config: AnnealedProxyEntropySamplerConfig):
        super().__init__(config)

    def proxy_weight(self) -> float:
        return self.config.initial_proxy_weight * exp(
            log(self.config.reference_fraction) * self.epoch / self.config.reference_epoch
        )

    def _uniform_for(self, question_id: object) -> float:
        payload = f"{self.config.seed}:{self.epoch}:{question_id}".encode()
        integer = int.from_bytes(blake2b(payload, digest_size=8).digest(), "big")
        # Clamp after float conversion: values near 2**64 can otherwise round to exactly 1.0.
        return min(max((integer + 1) / 2**64, 1e-12), 1.0 - 1e-12)

    @override
    def _score_row(self, row: dict) -> float:
        try:
            student_entropy = float(row["entropy_value"])
            proxy_entropy = float(row["teacher_entropy"])
        except (TypeError, ValueError):
            return float("nan")
        if not isfinite(student_entropy) or not isfinite(proxy_entropy):
            return float("nan")

        gain = max(student_entropy - self.proxy_weight() * proxy_entropy, 0.0)
        if gain <= 0:
            return gain

        if "question_id" not in row:
            raise KeyError(
                "AnnealedProxyEntropySampler requires question_id for deterministic sampling"
            )
        return gain / -log(self._uniform_for(row["question_id"]))
