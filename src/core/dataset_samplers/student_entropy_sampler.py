from typing import override

from core.dataset_samplers.abstract_sampler import AbstractDatasetSampler, AbstractDatasetSamplerConfig


class StudentEntropySamplerConfig(AbstractDatasetSamplerConfig):
    pass


class StudentEntropySampler(AbstractDatasetSampler):
    @override
    def _score_row(self, row: dict) -> float:
        return row["student_entropy"]
