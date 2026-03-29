from abc import abstractmethod
from typing import override

import pandas as pd
from datasets import Dataset, load_dataset
from pydantic import BaseModel

from core.dataset_samplers.base_sampler import BaseDatasetSampler
from core.datasets.abstract_dataset_adapter import AbstractDatasetAdapter
from core.datasets.base_dataset import BaseDataset


class TokenizedRow(BaseModel):
    model_config = {"extra": "allow"}

    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    row_id: str


class BaseDatasetAdapter[D: BaseDataset](AbstractDatasetAdapter):
    def __init__(self, dataset: D, dataset_sampler: BaseDatasetSampler | None = None):
        self.dataset = dataset
        self.dataset_sampler = dataset_sampler

    @abstractmethod
    def process_row(self, row: dict) -> TokenizedRow: ...

    def _load_ds(self, dataset: BaseDataset) -> Dataset:
        ds = load_dataset(
            "parquet",
            data_files={"default": dataset.processed_path},
        )
        return ds["default"]

    @override
    def process_dataset(self, path_override: str | None = None, strict: bool = True) -> Dataset:
        if path_override is not None:
            ds = self._load_ds(
                self.dataset.__class__(
                    config=self.dataset.config.model_copy(update={"path": path_override, "dataset_id": "tmp"}),
                    tokenizer=self.dataset.tokenizer,
                )
            )
        else:
            ds = self._load_ds(self.dataset)

        if self.dataset_sampler is not None:
            ds = self.dataset_sampler.create_sample(ds)

        ds = ds.map(
            lambda row: self.process_row(row).model_dump(),
            num_proc=4,
            remove_columns=ds.column_names if strict else None,
        )

        return ds

    @override
    def save_processed_dataset(self, df: pd.DataFrame, path: str, tmp: bool) -> None:
        df.to_parquet(path=path, index=False, compression=None if tmp else "snappy")
