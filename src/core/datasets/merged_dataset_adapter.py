from pathlib import Path
from typing import override

import pandas as pd
from datasets import Dataset, concatenate_datasets

from core.datasets.abstract_dataset_adapter import AbstractDatasetAdapter
from core.datasets.base_dataset_adapter import BaseDatasetAdapter


class MergedDatasetAdapter(AbstractDatasetAdapter):
    def __init__(self, dataset_adapters: list[BaseDatasetAdapter]):
        self.dataset_adapters = dataset_adapters

    @override
    def process_dataset(self, path_override: str | None = None) -> Dataset:
        datasets = [
            adapter.process_dataset(
                path_override=Path(path_override) / adapter.dataset.id if path_override is not None else None,
                strict=True,
            )
            for adapter in self.dataset_adapters
        ]

        ds = concatenate_datasets(datasets)

        return ds

    @override
    def save_processed_dataset(self, df: pd.DataFrame, path: str, tmp: bool) -> None:
        raise NotImplementedError(
            "Saving is not implemented for MergedDatasetAdapter. Please save individual datasets separately."
        )
