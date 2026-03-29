from abc import ABC, abstractmethod

import pandas as pd
from datasets import Dataset


class AbstractDatasetAdapter(ABC):
    @abstractmethod
    def process_dataset(self, path_override: str | None = None) -> Dataset: ...

    @abstractmethod
    def save_processed_dataset(self, df: pd.DataFrame, path: str, tmp: bool) -> None: ...
