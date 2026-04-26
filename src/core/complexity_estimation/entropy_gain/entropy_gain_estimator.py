from typing import override

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.tokenization_utils import PreTrainedTokenizer

from core.complexity_estimation.complexity_estimator import BaseComplexityEstimator
from core.complexity_estimation.entropy.logit_entropy import compute_entropy_from_logits
from core.datasets.base_dataset_adapter import TokenizedRow


class EntropyGainEstimatorSchema(BaseModel):
    student_entropy_value: float
    entropy_gain_value: float


class EntropyGainEstimator(BaseComplexityEstimator[EntropyGainEstimatorSchema]):
    def __init__(self, proxy_entropy_field_name: str) -> None:
        super().__init__()

        self.proxy_entropy_field_name = proxy_entropy_field_name

    @property
    @override
    def schema(self) -> dict[str, FieldInfo]:
        return EntropyGainEstimatorSchema.model_fields

    @override
    def estimate_row(
        self,
        dataset_row: dict,
        input: TokenizedRow,
        outputs: GenerateDecoderOnlyOutput,
        parsed_answer: str,
        answer_correctness: bool,
        tokenizer: PreTrainedTokenizer,
    ) -> EntropyGainEstimatorSchema:
        first_token_logits = outputs.scores[0][0]
        student_entropy = compute_entropy_from_logits(first_token_logits)
        proxy_entropy = dataset_row.get(self.proxy_entropy_field_name)
        entropy_gain = max(student_entropy - proxy_entropy, 0)
        return EntropyGainEstimatorSchema(
            student_entropy_value=student_entropy.item(), entropy_gain_value=entropy_gain.item()
        )
