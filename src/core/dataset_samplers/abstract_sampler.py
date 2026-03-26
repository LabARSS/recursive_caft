from abc import ABC

from datasets import Dataset
from pydraconf import PydraConfig

from core.datasets.abstract_dataset_adapter import abstractmethod


class AbstractDatasetSamplerConfig(PydraConfig):
    top_k: int


class AbstractDatasetSampler(ABC):
    def __init__(self, config: AbstractDatasetSamplerConfig):
        self.config = config

    @abstractmethod
    def _score_row(self, row: dict) -> float: ...

    def create_sample(self, ds: Dataset) -> Dataset:
        df = ds.to_pandas()
        df["score"] = df.apply(self._score_row, axis=1)
        df = df.sort_values("score", ascending=False)

        sampled_df = df.head(self.config.top_k)

        sampled_ds = Dataset.from_pandas(sampled_df)
        return sampled_ds
