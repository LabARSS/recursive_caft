"""BUG B regression: the resampling train dataset must be sized.

`ResamplingDataset` is a torch IterableDataset. HF's Trainer (transformers 4.52.3)
raises in `Trainer.__init__` ("does not implement __len__, max_steps has to be
specified") when the train dataset has no length and max_steps is unset — see
`has_length` at trainer.py:708. `ResamplingDataset.__len__` returns the sampler's
top_k so that check passes and the normal per-epoch loop runs. These tests assert
the exact condition that gates that error, with no model/GPU.
"""

from pathlib import Path

from datasets import Dataset
from transformers.trainer_utils import has_length

from core.datasets.abstract_dataset_adapter import AbstractDatasetAdapter
from core.complexity_estimation.complexity_estimation_runner import ModelGenerateConfig
from core.complexity_estimation.entropy.single_token_entropy_estimator import SingleTokenEntropyEstimator
from core.dataset_samplers.base_sampler import BaseDatasetSamplerConfig
from core.dataset_samplers.entropy_gain_sampler import EntropyGainSampler
from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.merged_dataset_adapter import MergedDatasetAdapter
from core.datasets.mmlu.mmlu_reasoning_response_dataset import MMLUReasoningResponseDataset
from core.datasets.mmlu.mmlu_single_token_response_dataset import MMLUSingleTokenResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.training.resampling_trainer import EstimateComplexityCallback, ResamplingDataset

TOP_K = 7


def _adapter(tokenizer, sampler):
    return CausalDatasetAdapter(
        dataset=MMLUReasoningResponseDataset(
            config=QADatasetConfig(path="sentinel-overridden-at-runtime", dataset_id="t"),
            tokenizer=tokenizer,
        ),
        dataset_sampler=sampler,
    )


def test_len_returns_sampler_top_k(thinking_tokenizer):
    adapter = _adapter(thinking_tokenizer, EntropyGainSampler(BaseDatasetSamplerConfig(top_k=TOP_K)))
    ds = ResamplingDataset(adapter, thinking_tokenizer)
    assert len(ds) == TOP_K


def test_has_length_true(thinking_tokenizer):
    # This is precisely the guard HF evaluates at trainer.py:708.
    adapter = _adapter(thinking_tokenizer, EntropyGainSampler(BaseDatasetSamplerConfig(top_k=TOP_K)))
    ds = ResamplingDataset(adapter, thinking_tokenizer)
    assert has_length(ds) is True


def test_len_does_no_io(thinking_tokenizer):
    # __len__ is evaluated in Trainer.__init__ before any on_epoch_begin sets the
    # path, so it must not read the (possibly absent) resampling parquet.
    adapter = _adapter(thinking_tokenizer, EntropyGainSampler(BaseDatasetSamplerConfig(top_k=TOP_K)))
    ds = ResamplingDataset(adapter, thinking_tokenizer)
    ds.dataset_path = "/nonexistent/path/should/not/be/read.parquet"
    assert len(ds) == TOP_K


def test_len_fallback_without_sampler(thinking_tokenizer, tmp_path, make_resampling_df):
    # No sampler: length falls back to the underlying (path-overridden) dataset size.
    parquet = tmp_path / "ds.parquet"
    make_resampling_df(3).to_parquet(parquet)
    adapter = _adapter(thinking_tokenizer, None)
    ds = ResamplingDataset(adapter, thinking_tokenizer)
    ds.dataset_path = str(parquet)
    assert len(ds) == 3


def test_len_over_merged_adapter(thinking_tokenizer):
    # A MergedDatasetAdapter reports the summed top_k of its sub-adapters, so HF still sees a
    # sized dataset (no max_steps error) when training on multiple datasets.
    merged = MergedDatasetAdapter(
        [
            _adapter(thinking_tokenizer, EntropyGainSampler(BaseDatasetSamplerConfig(top_k=7))),
            _adapter(thinking_tokenizer, EntropyGainSampler(BaseDatasetSamplerConfig(top_k=3))),
        ]
    )
    ds = ResamplingDataset(merged, thinking_tokenizer)
    assert has_length(ds) is True
    assert len(ds) == 10


def test_save_processed_dataset_creates_parent_dir(thinking_tokenizer, tmp_path):
    # The per-epoch complexity-estimation output lands in a fresh nested dir that
    # nothing created yet; the adapter must mkdir before writing.
    import pandas as pd

    adapter = _adapter(thinking_tokenizer, None)
    out = tmp_path / "0" / "complexity_estimation" / "x.parquet"
    assert not out.parent.exists()

    adapter.save_processed_dataset(pd.DataFrame({"x": [1, 2]}), path=str(out), tmp=True)

    assert out.exists()


class _EmptyAdapter(AbstractDatasetAdapter):
    """Minimal adapter whose post-sampling dataset is empty, to exercise __iter__'s guard."""

    def process_dataset(self, path_override=None, shuffle=False, shuffle_seed=None):
        return Dataset.from_dict({"input_ids": [], "attention_mask": [], "labels": [], "row_id": []})

    def save_processed_dataset(self, df, path, tmp):
        raise NotImplementedError

    def override_tokenizer(self, tokenizer):
        pass


class _EpochRecordingAdapter(_EmptyAdapter):
    def __init__(self):
        self.seen_epochs = []

    def set_epoch(self, epoch):
        self.seen_epochs.append(epoch)


def test_iter_does_not_crash_on_empty_dataset(thinking_tokenizer):
    # The one-time sample logging does dataset[0]; on an empty post-filter dataset that must not
    # raise IndexError before yielding (the guard `len(dataset) > 0`).
    ds = ResamplingDataset(_EmptyAdapter(), thinking_tokenizer)
    assert list(ds) == []


def test_iter_sets_effective_selection_epoch():
    adapter = _EpochRecordingAdapter()
    ds = ResamplingDataset(adapter, tokenizer=None)  # type: ignore[arg-type]
    ds.selection_epoch = 17

    assert list(ds) == []
    assert adapter.seen_epochs == [17]


def test_shuffle_is_deterministic_within_epoch_and_changes_across_epochs(
    thinking_tokenizer, tmp_path, make_resampling_df
):
    parquet = tmp_path / "ds.parquet"
    source = make_resampling_df(8)
    source.to_parquet(parquet)
    adapter = _adapter(thinking_tokenizer, None)
    ds = ResamplingDataset(adapter, thinking_tokenizer, shuffle=True, data_seed=123)
    ds.dataset_path = str(parquet)

    ds.epoch = 4
    first = [row["row_id"] for row in ds]
    replay = [row["row_id"] for row in ds]
    ds.epoch = 5
    next_epoch = [row["row_id"] for row in ds]

    assert first == replay
    assert first != next_epoch
    assert set(first) == set(next_epoch) == set(source["question_id"])


def test_out_path_for_epoch_uses_dataset_id(thinking_tokenizer):
    # The per-epoch complexity-estimation path is built from the complexity
    # dataset's dataset_id; it must not reference a nonexistent `config.id`.
    callback = EstimateComplexityCallback(
        complexity_evaluation_dataset=QADatasetAdapter(
            dataset=MMLUSingleTokenResponseDataset(
                config=QADatasetConfig(path="sentinel", dataset_id="mmlu_teacher_entropy"),
                tokenizer=thinking_tokenizer,
            )
        ),
        complexity_estimator=SingleTokenEntropyEstimator(),
        complexity_estimation_runner_generation_config=ModelGenerateConfig(max_new_tokens=1),
        out_path=Path("/tmp/resampling_trainer_data"),
    )

    path = callback.out_path_for_epoch(3)

    assert path == Path("/tmp/resampling_trainer_data/3/complexity_estimation/mmlu_teacher_entropy.parquet")
