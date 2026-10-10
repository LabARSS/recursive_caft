from abc import ABC, abstractmethod

import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizer


class AbstractDatasetAdapter(ABC):
    @abstractmethod
    def process_dataset(
        self,
        path_override: str | None = None,
        shuffle: bool = False,
        shuffle_seed: int | None = None,
    ) -> Dataset: ...

    @abstractmethod
    def save_processed_dataset(self, df: pd.DataFrame, path: str, tmp: bool) -> None: ...

    def sampled_size(self) -> int | None:
        """Row count after sampling without materializing, or None when it can't be known
        without loading the source (e.g. no sampler)."""
        return None

    def selected_sample_counts(self, df: pd.DataFrame) -> dict[str, int]:
        """How many rows each sampler would actually select from `df`, keyed by dataset_id.
        Empty when there is no sampler (nothing to report)."""
        return {}

    def set_epoch(self, epoch: int) -> None:
        """Forward the effective resampling epoch to any sampler owned by the adapter."""
        return None

    @abstractmethod
    def override_tokenizer(self, tokenizer: PreTrainedTokenizer) -> None: ...
