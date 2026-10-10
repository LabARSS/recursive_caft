from math import log
from random import random
from typing import override

from core.dataset_samplers.base_sampler import BaseDatasetSampler

# -log(u) must stay finite and strictly positive: u == 0 has no exponential variate, and u == 1
# would make the key infinite. random() returns [0, 1), so only the lower end needs a floor.
U_FLOOR = 1e-12


class StudentEntropyProportionalSampler(BaseDatasetSampler):
    @override
    def _score_row(self, row: dict) -> float:
        student_entropy = row["entropy_value"]

        u = float(row["random_value"])
        if not 0.0 < u < 1.0:
            u = random()

        return student_entropy / -log(max(u, U_FLOOR))
