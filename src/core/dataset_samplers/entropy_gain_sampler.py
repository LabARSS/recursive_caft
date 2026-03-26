from typing import override

from core.dataset_samplers.abstract_sampler import AbstractDatasetSampler, AbstractDatasetSamplerConfig


class TeacherEntropySamplerConfig(AbstractDatasetSamplerConfig):
    pass


class EntropyGainSampler(AbstractDatasetSampler):
    @override
    def _score_row(self, row: dict) -> float:
        return max(row["student_entropy"] - row["teacher_entropy"], 0)
