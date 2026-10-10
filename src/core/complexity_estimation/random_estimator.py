from random import random
from typing import override

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.tokenization_utils import PreTrainedTokenizer

from core.complexity_estimation.complexity_estimator import BaseComplexityEstimator
from core.datasets.base_dataset_adapter import TokenizedRow


class RandomEstimatorSchema(BaseModel):
    random_value: float


class RandomEstimator(BaseComplexityEstimator[RandomEstimatorSchema]):
    @property
    @override
    def schema(self) -> dict[str, FieldInfo]:
        return RandomEstimatorSchema.model_fields

    @override
    def estimate_row(
        self,
        dataset_row: dict,
        input: TokenizedRow,
        outputs: GenerateDecoderOnlyOutput,
        parsed_answer: str,
        answer_correctness: bool,
        tokenizer: PreTrainedTokenizer,
    ) -> RandomEstimatorSchema:
        return RandomEstimatorSchema(random_value=random())

    @property
    @override
    def verify_model_output(self) -> bool:
        return False
