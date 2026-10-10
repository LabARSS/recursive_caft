from typing import override

from core.dataset_samplers.base_sampler import BaseDatasetSampler, BaseDatasetSamplerConfig


class TeacherEntropySamplerConfig(BaseDatasetSamplerConfig):
    pass


class EntropyGainSampler(BaseDatasetSampler):
    _EPS = 1e-8

    @override
    def _score_row(self, row: dict) -> float:
        return row["entropy_value"] / (row["teacher_entropy"] + self._EPS)
