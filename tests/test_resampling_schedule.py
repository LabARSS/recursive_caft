"""`resampling_schedule` freezes the training sample between scheduled epochs.

Default (None) = today's behaviour: every epoch re-estimates complexity and re-samples. A list
restricts (re)sampling to those epochs -- [0] draws once, on the untrained student, and trains on
that set for the whole run. Frozen epochs must skip estimation entirely (that is the whole point:
no weight snapshot, no estimation subprocess) while still pointing the dataset at the last
scheduled epoch's parquet, including when a resume makes a frozen epoch the FIRST on_epoch_begin.
"""

import json
import types
from math import nan

import pandas as pd
import pytest
from pydantic import ValidationError

from core.complexity_estimation.complexity_estimation_runner import ModelGenerateConfig
from core.complexity_estimation.entropy.single_token_entropy_estimator import SingleTokenEntropyEstimator
from core.dataset_samplers.base_sampler import BaseDatasetSamplerConfig
from core.dataset_samplers.entropy_gain_sampler import EntropyGainSampler
from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.mmlu.mmlu_reasoning_response_dataset import MMLUReasoningResponseDataset
from core.datasets.mmlu.mmlu_single_token_response_dataset import MMLUSingleTokenResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.training.lora_trainer import LoRATrainingArgs
from core.training.resampling_trainer import (
    EstimateComplexityCallback,
    ResamplingTrainerConfig,
    SetResamplingPathCallback,
)


def _callback(out_path, resampling_schedule=None, save_schedule=None):
    # The schedule predicates only touch out_path / dataset_id / the two schedules and the marker
    # files, so a lightweight stand-in for the dataset adapter is enough (no tokenizer/model).
    return EstimateComplexityCallback(
        complexity_evaluation_dataset=types.SimpleNamespace(
            dataset=types.SimpleNamespace(dataset_id="mmlu_teacher_entropy")
        ),
        complexity_estimator=None,
        complexity_estimation_runner_generation_config=None,
        out_path=out_path,
        save_schedule=save_schedule,
        resampling_schedule=resampling_schedule,
    )


def _write_epoch(callback, epoch, entropy_values):
    path = callback.out_path_for_epoch(epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"question_id": list(range(len(entropy_values))), "entropy_value": entropy_values}
    ).to_parquet(path, index=False)
    return path


# --- resampling_epoch_for -------------------------------------------------------------------


def test_no_schedule_resamples_every_epoch(tmp_path):
    cb = _callback(tmp_path, resampling_schedule=None)
    for epoch in (0, 1, 7, 100):
        assert cb.resampling_epoch_for(epoch) == epoch


def test_sample_once_pins_every_epoch_to_zero(tmp_path):
    cb = _callback(tmp_path, resampling_schedule=[0])
    for epoch in (0, 1, 7, 100):
        assert cb.resampling_epoch_for(epoch) == 0


def test_sparse_schedule_uses_latest_scheduled_epoch(tmp_path):
    cb = _callback(tmp_path, resampling_schedule=[0, 10, 20])
    assert cb.resampling_epoch_for(0) == 0
    assert cb.resampling_epoch_for(9) == 0
    assert cb.resampling_epoch_for(10) == 10
    assert cb.resampling_epoch_for(19) == 10
    assert cb.resampling_epoch_for(20) == 20
    # Past the end of the schedule the last draw simply stays frozen.
    assert cb.resampling_epoch_for(25) == 20


# --- on_epoch_begin skipping ----------------------------------------------------------------


def test_frozen_epoch_skips_estimation(tmp_path):
    # No kwargs["model"] is passed: on the estimating branch that raises KeyError (see
    # test_complexity_reuse_gating), so returning None proves the frozen epoch short-circuited
    # before any snapshot save or subprocess spawn.
    cb = _callback(tmp_path, resampling_schedule=[0])
    assert cb.on_epoch_begin(None, types.SimpleNamespace(epoch=3.0), None) is None


def test_scheduled_epoch_still_estimates(tmp_path):
    cb = _callback(tmp_path, resampling_schedule=[0, 10])
    for epoch in (0.0, 10.0):
        with pytest.raises(KeyError):
            cb.on_epoch_begin(None, types.SimpleNamespace(epoch=epoch), None)


def test_no_schedule_estimates_every_epoch(tmp_path):
    # Regression guard: the default path must be untouched by the new early return.
    cb = _callback(tmp_path, resampling_schedule=None)
    with pytest.raises(KeyError):
        cb.on_epoch_begin(None, types.SimpleNamespace(epoch=3.0), None)


# --- SetResamplingPathCallback --------------------------------------------------------------


def _set_path(cb, epoch):
    ds = types.SimpleNamespace(dataset_path=None, selection_epoch=None)
    SetResamplingPathCallback(estimation_complexity_callback=cb, resampling_ds=ds).on_epoch_begin(
        None, types.SimpleNamespace(epoch=float(epoch)), None
    )
    return ds.dataset_path


def test_frozen_epochs_keep_reading_epoch_zero(tmp_path):
    cb = _callback(tmp_path, resampling_schedule=[0])
    expected = cb.out_path_for_epoch(0).as_posix()
    for epoch in (0, 1, 2, 99):
        assert _set_path(cb, epoch) == expected


def test_resume_into_a_frozen_epoch_resolves_the_path(tmp_path):
    # A run restarted at epoch 7 never sees on_epoch_begin(0), so "latest scheduled <= epoch" --
    # not "did this epoch resample" -- is what keeps dataset_path from staying None.
    cb = _callback(tmp_path, resampling_schedule=[0])
    assert _set_path(cb, 7) == cb.out_path_for_epoch(0).as_posix()


def test_no_schedule_path_tracks_the_epoch(tmp_path):
    cb = _callback(tmp_path, resampling_schedule=None)
    assert _set_path(cb, 4) == cb.out_path_for_epoch(4).as_posix()


def test_set_path_callback_records_training_epoch_for_shuffle(tmp_path):
    cb = _callback(tmp_path, resampling_schedule=[0])
    ds = types.SimpleNamespace(dataset_path=None, epoch=None, selection_epoch=None)
    SetResamplingPathCallback(estimation_complexity_callback=cb, resampling_ds=ds).on_epoch_begin(
        None, types.SimpleNamespace(epoch=7.0), None
    )

    assert ds.epoch == 7
    assert ds.selection_epoch == 0
    assert ds.dataset_path == cb.out_path_for_epoch(0).as_posix()


def test_set_path_callback_uses_effective_epoch_for_selection(tmp_path):
    cb = _callback(tmp_path, resampling_schedule=[0, 10, 20])
    ds = types.SimpleNamespace(dataset_path=None, epoch=None, selection_epoch=None)
    SetResamplingPathCallback(estimation_complexity_callback=cb, resampling_ds=ds).on_epoch_begin(
        None, types.SimpleNamespace(epoch=17.0), None
    )

    assert ds.epoch == 17
    assert ds.selection_epoch == 10
    assert ds.dataset_path == cb.out_path_for_epoch(10).as_posix()


# --- schedule-aware backfill ----------------------------------------------------------------


def test_sparse_backfill_skips_back_to_the_last_dump(tmp_path):
    # Under [0, 10] epochs 1..9 wrote no parquet at all, so epoch 10 must backfill from epoch 0
    # rather than blowing up reading a nonexistent epoch 9.
    cb = _callback(tmp_path, resampling_schedule=[0, 10])
    _write_epoch(cb, 0, [0.5] * 20)
    epoch10 = [0.9] * 20
    epoch10[3] = nan
    path = _write_epoch(cb, 10, epoch10)

    cb._reconcile_estimation(10)

    assert pd.read_parquet(path)["entropy_value"][3] == 0.5
    stats = json.loads(cb._stats_path_for_epoch(10).read_text())
    assert stats["backfilled"] == 1
    assert stats["residual_failed"] == 0


def test_sample_once_epoch_zero_has_no_backfill(tmp_path):
    # [0] only ever reconciles epoch 0, which has no predecessor: NaNs stay NaN (the sampler drops
    # them) and nothing tries to read epoch -1.
    cb = _callback(tmp_path, resampling_schedule=[0])
    values = [0.9] * 20
    values[3] = nan
    path = _write_epoch(cb, 0, values)

    cb._reconcile_estimation(0)

    assert pd.isna(pd.read_parquet(path)["entropy_value"][3])
    assert json.loads(cb._stats_path_for_epoch(0).read_text())["backfilled"] == 0


# --- config validation ----------------------------------------------------------------------


def _config(thinking_tokenizer, resampling_schedule, tmp_path):
    return ResamplingTrainerConfig(
        out_path=tmp_path.as_posix(),
        model_id="sentinel-model",
        training_args=LoRATrainingArgs(num_train_epochs=10, per_device_train_batch_size=2),
        train_dataset=CausalDatasetAdapter(
            dataset=MMLUReasoningResponseDataset(
                config=QADatasetConfig(path="sentinel", dataset_id="train"),
                tokenizer=thinking_tokenizer,
            ),
            dataset_sampler=EntropyGainSampler(BaseDatasetSamplerConfig(top_k=8)),
        ),
        complexity_evaluation_dataset=QADatasetAdapter(
            dataset=MMLUSingleTokenResponseDataset(
                config=QADatasetConfig(path="sentinel", dataset_id="mmlu_teacher_entropy"),
                tokenizer=thinking_tokenizer,
            )
        ),
        complexity_estimator=SingleTokenEntropyEstimator(),
        complexity_estimation_runner_generation_config=ModelGenerateConfig(max_new_tokens=1),
        resampling_schedule=resampling_schedule,
    )


@pytest.mark.parametrize("schedule", [None, [0], [0, 10, 20]])
def test_config_accepts_valid_schedules(thinking_tokenizer, tmp_path, schedule):
    assert _config(thinking_tokenizer, schedule, tmp_path).resampling_schedule == schedule


@pytest.mark.parametrize(
    "schedule",
    [
        [],  # empty: None is how you ask for every-epoch resampling
        [1, 2],  # does not start at 0 -> epoch 0 would have no parquet to read
        [0, 5, 5],  # duplicate
        [0, 20, 10],  # unsorted
    ],
)
def test_config_rejects_invalid_schedules(thinking_tokenizer, tmp_path, schedule):
    with pytest.raises(ValidationError):
        _config(thinking_tokenizer, schedule, tmp_path)
