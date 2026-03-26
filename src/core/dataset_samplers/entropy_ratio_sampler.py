from typing import override

from core.dataset_samplers.abstract_sampler import AbstractDatasetSampler, AbstractDatasetSamplerConfig


class TeacherEntropySamplerConfig(AbstractDatasetSamplerConfig):
    pass


class EntropyGainSampler(AbstractDatasetSampler):
    _EPS = 1e-8

    @override
    def _score_row(self, row: dict) -> float:
        return row["student_entropy"] / (row["teacher_entropy"] + self._EPS)
