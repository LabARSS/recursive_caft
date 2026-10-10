from abc import ABC, abstractmethod

from core.datasets.causal_dataset import CausalDataset, CausalDatasetConfig


class InvalidAnswerError(Exception):
    """Raised by verify_assistant_response when the model did not produce a parseable answer
    (e.g. an empty single-token response because it emitted a thinking/special token). Callers
    treat this as an expected failed measurement rather than an unexpected error."""


class QADatasetConfig(CausalDatasetConfig):
    pass


class QADataset[C: QADatasetConfig](CausalDataset[C], ABC):
    @abstractmethod
    def verify_assistant_response(self, row: dict, assistant_response: str) -> tuple[str, bool]: ...
