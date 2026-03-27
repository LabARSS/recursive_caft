from abc import ABC, abstractmethod

import pandas as pd
from pydraconf import PydraConfig

from core.utils.logger import logger


class BaseDatasetAggregatorConfig(PydraConfig):
    in_paths: str
    out_path: str


class BaseDatasetAggregator(ABC):
    def __init__(self, config: BaseDatasetAggregatorConfig):
        self.config = config

    @abstractmethod
    def _merge(self, dfs: list[pd.DataFrame]) -> pd.DataFrame: ...

    def aggregate(self) -> None:
        logger.info(f"Aggregating datasets from {self.config.in_paths} into {self.config.out_path}...")
        dfs = [pd.read_parquet(path) for path in self.config.in_paths]
        merged_df = self._merge(dfs)
        merged_df.to_parquet(self.config.out_path)
