from typing import override

from core.dataset_samplers.base_sampler import BaseDatasetSampler, BaseDatasetSamplerConfig


class TeacherEntropySamplerConfig(BaseDatasetSamplerConfig):
    pass


class TeacherEntropySampler(BaseDatasetSampler):
    @override
    def _score_row(self, row: dict) -> float:
        return row["teacher_entropy"]
