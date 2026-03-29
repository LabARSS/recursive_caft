from typing import override

from core.dataset_samplers.base_sampler import BaseDatasetSampler


class StudentEntropySampler(BaseDatasetSampler):
    @override
    def _score_row(self, row: dict) -> float:
        return row["student_entropy"]
