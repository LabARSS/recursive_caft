from abc import ABC, abstractmethod

from pydraconf import PydraConfig
from transformers import PreTrainedTokenizer

from core.datasets.base_dataset_aggregator import BaseDatasetAggregator


class BaseDatasetConfig(PydraConfig):
    path: str | BaseDatasetAggregator
    dataset_id: str


class BaseDataset[C: BaseDatasetConfig](ABC):
    def __init__(self, tokenizer: PreTrainedTokenizer, config: C):
        self.tokenizer = tokenizer
        self.config = config
        self._path = config.path if isinstance(config.path, str) else None

    @abstractmethod
    def system_prompt(self, row: dict) -> str: ...

    @abstractmethod
    def user_prompt(self, row: dict) -> str: ...

    @abstractmethod
    def row_id(self, row: dict) -> str: ...

    @property
    def dataset_id(self) -> str:
        return self.config.dataset_id

    @property
    def processed_path(self) -> str:
        if self._path is None:
            assert isinstance(self.config.path, BaseDatasetAggregator)
            self._path = self.config.path.aggregate()

        assert self._path is not None
        return self._path
