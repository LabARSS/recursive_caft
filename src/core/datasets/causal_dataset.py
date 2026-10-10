from abc import ABC, abstractmethod

from core.datasets.base_dataset import BaseDataset, BaseDatasetConfig


class CausalDatasetConfig(BaseDatasetConfig):
    pass


class CausalDataset[C: CausalDatasetConfig](BaseDataset[C], ABC):
    @abstractmethod
    def assistant_response(self, row: dict) -> str: ...
