from abc import abstractmethod
from pathlib import Path
from typing import override

import pandas as pd
from datasets import Dataset, load_dataset
from pydantic import BaseModel
from transformers import PreTrainedTokenizer

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

    def sampled_size(self) -> int | None:
        """Row count after sampling, without materializing the dataset. Returns the sampler's
        top_k, or None when there is no sampler (size is only known by loading the source)."""
        return self.dataset_sampler.config.top_k if self.dataset_sampler is not None else None

    @override
    def selected_sample_counts(self, df: pd.DataFrame) -> dict[str, int]:
        if self.dataset_sampler is None:
            return {}
        return {self.dataset.dataset_id: self.dataset_sampler.count_selected(df)}

    @override
    def set_epoch(self, epoch: int) -> None:
        if self.dataset_sampler is not None:
            self.dataset_sampler.set_epoch(epoch)

    def _load_ds(self, dataset: BaseDataset) -> Dataset:
        ds = load_dataset(
            "parquet",
            data_files={"default": dataset.processed_path},
        )
        return ds["default"]

    @override
    def process_dataset(
        self,
        path_override: str | None = None,
        strict: bool = True,
        shuffle: bool = False,
        shuffle_seed: int | None = None,
    ) -> Dataset:
        if path_override is not None:
            ds = self._load_ds(
                self.dataset.__class__(
                    config=self.dataset.config.model_copy(update={"path": path_override}),
                    tokenizer=self.dataset.tokenizer,
                )
            )
        else:
            ds = self._load_ds(self.dataset)

        if self.dataset_sampler is not None:
            ds = self.dataset_sampler.create_sample(ds)

        # A re-processed dump may already carry tokenization columns produced by an earlier
        # process_row pass (e.g. the complexity runner's .tmp resume file, which keeps them until its
        # final save). process_row re-tokenizes and passes these as explicit kwargs, so leaving them
        # in the row collides with **row ("multiple values for keyword argument"). Drop them first;
        # they are always regenerated below, so this is a no-op for source / final-parquet inputs.
        stale_columns = [c for c in ("input_ids", "attention_mask", "labels", "row_id") if c in ds.column_names]
        if stale_columns:
            ds = ds.remove_columns(stale_columns)

        ds = ds.map(
            lambda row: self.process_row(row).model_dump(),
            num_proc=4,
            remove_columns=ds.column_names if strict else None,
        )

        if shuffle:
            ds = ds.shuffle(seed=shuffle_seed)

        return ds

    @override
    def save_processed_dataset(self, df: pd.DataFrame, path: str, tmp: bool) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path=path, index=False, compression=None if tmp else "snappy")

    @override
    def override_tokenizer(self, tokenizer: PreTrainedTokenizer) -> None:
        self.dataset.tokenizer = tokenizer
