from typing import override

from core.dataset_samplers.base_sampler import BaseDatasetSampler, BaseDatasetSamplerConfig


class TeacherEntropySamplerConfig(BaseDatasetSamplerConfig):
    pass


class EntropyGainSampler(BaseDatasetSampler):
    @override
    def _score_row(self, row: dict) -> float:
        return max(row["student_entropy"] - row["teacher_entropy"], 0)
