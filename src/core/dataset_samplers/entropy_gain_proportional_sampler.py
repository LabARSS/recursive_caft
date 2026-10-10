from math import log
from random import random
from typing import override

from core.dataset_samplers.base_sampler import BaseDatasetSampler

# -log(u) must stay finite and strictly positive: u == 0 has no exponential variate, and u == 1
# would make the key infinite. random() returns [0, 1), so only the lower end needs a floor.
U_FLOOR = 1e-12


class EntropyGainProportionalSampler(BaseDatasetSampler):
    """Samples top_k rows *at random with probability proportional to entropy gain*, rather than
    taking the top_k highest-gain rows.

    It does this without touching the deterministic selection in BaseDatasetSampler: the score is a
    perturbed gain, an exponential race key `gain / E` with `E ~ Exp(1)` drawn per row per epoch
    (`E = -log(random_value)`, from SingleTokenEntropyWithRandomEstimator). Taking the top_k of that
    key is the Efraimidis-Spirakis weighted reservoir scheme, i.e. exactly a without-replacement
    sample where P(a given row is drawn first) = gain / sum(gain).

    The key is positive iff the gain is positive and NaN iff the gain is NaN, so `drop_non_positive`
    keeps its meaning: rows the student has already caught the teacher on, and rows whose entropy
    could not be measured, are still dropped before the draw.

    Note the trailing curriculum sort in `_select` then orders by the perturbed key, so the
    easy-to-hard ordering within an epoch is a noisy version of the gain ordering.
    """

    @override
    def _score_row(self, row: dict) -> float:
        gain = max(row["entropy_value"] - row["teacher_entropy"], 0)
        if gain <= 0:
            return float(gain)
        # An unmeasured row scores NaN: NaN <= 0 is False, so it falls through here and the NaN
        # propagates through the key below, which drop_non_positive then discards -- same as the
        # plain entropy-gain sampler.

        u = float(row["random_value"])
        if not 0.0 < u < 1.0:
            # random_value is NaN when the row's measurement failed this epoch. The trainer's
            # reconcile pass backfills only entropy_value, so redraw here to keep such rows as
            # eligible as they are for the plain entropy-gain sampler.
            u = random()

        return gain / -log(max(u, U_FLOOR))
