from typing import override

from core.dataset_samplers.abstract_sampler import AbstractDatasetSampler, AbstractDatasetSamplerConfig


class TeacherEntropySamplerConfig(AbstractDatasetSamplerConfig):
    pass


class TeacherEntropySampler(AbstractDatasetSampler):
    @override
    def _score_row(self, row: dict) -> float:
        return row["teacher_entropy"]
