from random import random
from typing import override

from pydantic.fields import FieldInfo
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.tokenization_utils import PreTrainedTokenizer

from core.complexity_estimation.entropy.single_token_entropy_estimator import (
    SingleTokenEntropyEstimator,
    SingleTokenEntropyEstimatorSchema,
)
from core.datasets.base_dataset_adapter import TokenizedRow


class SingleTokenEntropyWithRandomSchema(SingleTokenEntropyEstimatorSchema):
    random_value: float


class SingleTokenEntropyWithRandomEstimator(SingleTokenEntropyEstimator):
    """Single-token entropy plus a fresh uniform draw per row.

    The estimator is the only per-epoch hook: it is re-run on the whole pool every epoch and writes
    the parquet the sampler then scores. Emitting `random_value` here (the same column and source as
    RandomEstimator) is what gives a stochastic sampler a new draw each epoch without any state of
    its own — see EntropyGainProportionalSampler.

    Like RandomEstimator, the draw comes from the global `random` module inside the estimation
    subprocess, which never calls `set_seed()`, so it is not reproducible from a config value.
    """

    @property
    @override
    def schema(self) -> dict[str, FieldInfo]:
        return SingleTokenEntropyWithRandomSchema.model_fields

    @override
    def estimate_row(
        self,
        dataset_row: dict,
        input: TokenizedRow,
        outputs: GenerateDecoderOnlyOutput,
        parsed_answer: str,
        answer_correctness: bool,
        tokenizer: PreTrainedTokenizer,
    ) -> SingleTokenEntropyWithRandomSchema:
        entropy = super().estimate_row(dataset_row, input, outputs, parsed_answer, answer_correctness, tokenizer)
        return SingleTokenEntropyWithRandomSchema(entropy_value=entropy.entropy_value, random_value=random())
