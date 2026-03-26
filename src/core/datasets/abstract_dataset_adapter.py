from abc import ABC, abstractmethod

from datasets import Dataset, load_dataset, load_from_disk
from pydantic import BaseModel

from core.dataset_samplers.abstract_sampler import AbstractDatasetSampler
from core.datasets.base_dataset import BaseDataset


class TokenizedRow(BaseModel):
    model_config = {"extra": "allow"}

    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    row_id: str


class AbstractDatasetAdapter[D: BaseDataset](ABC):
    def __init__(self, dataset: D, dataset_sampler: AbstractDatasetSampler | None = None):
        self.dataset = dataset
        self.dataset_sampler = dataset_sampler

    @abstractmethod
    def process_row(self, row: dict) -> TokenizedRow: ...

    def _load_ds(self, path_override: str | None = None) -> Dataset:
        if path_override is not None:
            ds = load_from_disk(path_override)
            return ds

        ds = load_dataset(
            "parquet",
            data_files={"default": self.dataset.config.path},
        )
        return ds["default"]

    def process_dataset(self, path_override: str | None = None) -> Dataset:
        ds = self._load_ds(path_override)

        if self.dataset_sampler is not None:
            ds = self.dataset_sampler.create_sample(ds)

        processed_ds = ds.map(
            lambda row: self.process_row(row).model_dump(),
            num_proc=4,
            remove_columns=ds.column_names,
        )

        return processed_ds
