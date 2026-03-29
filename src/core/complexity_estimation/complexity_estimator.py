from abc import ABC, abstractmethod

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.tokenization_utils import PreTrainedTokenizer

from core.datasets.base_dataset_adapter import TokenizedRow


class BaseComplexityEstimator[T: BaseModel](ABC):
    @property
    @abstractmethod
    def schema(self) -> dict[str, FieldInfo]: ...

    @abstractmethod
    def estimate_row(
        self,
        dataset_row: dict,
        input: TokenizedRow,
        outputs: GenerateDecoderOnlyOutput,
        parsed_answer: str,
        answer_correctness: bool,
        tokenizer: PreTrainedTokenizer,
    ) -> T: ...
