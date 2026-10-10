from typing import override

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.tokenization_utils import PreTrainedTokenizer

from core.complexity_estimation.complexity_estimator import BaseComplexityEstimator
from core.complexity_estimation.entropy.logit_entropy import compute_entropy_from_logits
from core.datasets.base_dataset_adapter import TokenizedRow


class SingleTokenEntropyEstimatorSchema(BaseModel):
    entropy_value: float


class SingleTokenEntropyEstimator(BaseComplexityEstimator[SingleTokenEntropyEstimatorSchema]):
    @property
    @override
    def schema(self) -> dict[str, FieldInfo]:
        return SingleTokenEntropyEstimatorSchema.model_fields

    @override
    def estimate_row(
        self,
        dataset_row: dict,
        input: TokenizedRow,
        outputs: GenerateDecoderOnlyOutput,
        parsed_answer: str,
        answer_correctness: bool,
        tokenizer: PreTrainedTokenizer,
    ) -> SingleTokenEntropyEstimatorSchema:
        first_token_logits = outputs.scores[0][0]
        entropy = compute_entropy_from_logits(first_token_logits)
        return SingleTokenEntropyEstimatorSchema(entropy_value=entropy.item())
