"""EstimateComplexityCallback._reconcile_estimation: backfill rows that failed to measure
single-token entropy from the previous epoch's last-known-good values (joined on question_id).

An epoch whose failure rate exceeds max_failed_estimation_fraction is tolerated once -- the
backfill carries it -- and aborts training only if the next epoch exceeds the threshold too.
Epoch 0 has no previous epoch to backfill from, so it aborts immediately."""

import json
from math import nan
from types import SimpleNamespace

import pandas as pd
import pytest

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
from core.training.resampling_trainer import EstimateComplexityCallback, ResamplingTrainer


def _train_dataset(thinking_tokenizer, specs):
    # specs: list of (dataset_id, top_k). Builds train adapters whose EntropyGainSampler the
    # callback queries for selected_sample_counts. count_selected reads entropy_value/teacher_entropy
    # off the parquet directly — no tokenization — so the sentinel path is never loaded.
    return MergedDatasetAdapter(
        [
            CausalDatasetAdapter(
                dataset=MMLUReasoningResponseDataset(
                    config=QADatasetConfig(path="sentinel", dataset_id=dataset_id),
                    tokenizer=thinking_tokenizer,
                ),
                dataset_sampler=EntropyGainSampler(BaseDatasetSamplerConfig(top_k=top_k)),
            )
            for dataset_id, top_k in specs
        ]
    )


def _callback(thinking_tokenizer, out_path, max_failed=0.1, train_dataset=None):
    return EstimateComplexityCallback(
        complexity_evaluation_dataset=QADatasetAdapter(
            dataset=MMLUSingleTokenResponseDataset(
                config=QADatasetConfig(path="sentinel", dataset_id="mmlu_teacher_entropy"),
                tokenizer=thinking_tokenizer,
            )
        ),
        complexity_estimator=SingleTokenEntropyEstimator(),
        complexity_estimation_runner_generation_config=ModelGenerateConfig(max_new_tokens=1),
        out_path=out_path,
        max_failed_estimation_fraction=max_failed,
        train_dataset=train_dataset,
    )


def _write_epoch(callback, epoch, entropy_values, teacher=None):
    path = callback.out_path_for_epoch(epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"question_id": list(range(len(entropy_values))), "entropy_value": entropy_values}
    if teacher is not None:
        data["teacher_entropy"] = [teacher] * len(entropy_values)
    pd.DataFrame(data).to_parquet(path, index=False)
    return path


def _read_stats(callback, epoch):
    return json.loads(callback._stats_path_for_epoch(epoch).read_text())


def test_reconcile_backfills_from_previous_epoch(thinking_tokenizer, tmp_path):
    cb = _callback(thinking_tokenizer, tmp_path)
    n = 20
    _write_epoch(cb, 0, [0.5] * n)
    epoch1 = [0.9] * n
    epoch1[3] = nan  # one row failed this epoch (5% <= 10%)
    path1 = _write_epoch(cb, 1, epoch1)

    cb._reconcile_estimation(1)

    out = pd.read_parquet(path1)
    assert out["entropy_value"].notna().all()
    assert out.loc[out["question_id"] == 3, "entropy_value"].iloc[0] == 0.5  # from epoch 0
    assert out.loc[out["question_id"] == 0, "entropy_value"].iloc[0] == 0.9  # untouched


def test_reconcile_aborts_above_threshold(thinking_tokenizer, tmp_path):
    # Epoch 0 gets no grace epoch: there is no previous epoch to backfill the failed rows from.
    cb = _callback(thinking_tokenizer, tmp_path, max_failed=0.1)
    entropy = [0.5] * 10
    entropy[0] = nan
    entropy[1] = nan  # 2/10 = 20% > 10%
    _write_epoch(cb, 0, entropy)

    with pytest.raises(RuntimeError, match="no previous epoch to backfill from"):
        cb._reconcile_estimation(0)


def _failing_epoch(n=10, failed=3, value=0.9):
    # `failed`/`n` > the default max_failed=0.1 threshold.
    entropy = [value] * n
    for i in range(failed):
        entropy[i] = nan
    return entropy


def test_reconcile_tolerates_single_exceedance(thinking_tokenizer, tmp_path):
    cb = _callback(thinking_tokenizer, tmp_path, max_failed=0.1)
    _write_epoch(cb, 0, [0.5] * 10)
    cb._reconcile_estimation(0)
    path1 = _write_epoch(cb, 1, _failing_epoch())  # 30% > 10%

    cb._reconcile_estimation(1)  # grace epoch: no raise

    out = pd.read_parquet(path1)
    assert out["entropy_value"].notna().all()  # all three failures backfilled from epoch 0
    assert (out.loc[out["question_id"] < 3, "entropy_value"] == 0.5).all()
    stats = _read_stats(cb, 1)
    assert stats["exceeded_threshold"] is True
    assert stats["failed_rows"] == 3  # pre-backfill count
    assert stats["residual_failed"] == 0
    assert _read_stats(cb, 0)["exceeded_threshold"] is False


def test_reconcile_aborts_on_two_consecutive_exceedances(thinking_tokenizer, tmp_path):
    cb = _callback(thinking_tokenizer, tmp_path, max_failed=0.1)
    _write_epoch(cb, 0, [0.5] * 10)
    cb._reconcile_estimation(0)
    _write_epoch(cb, 1, _failing_epoch())
    cb._reconcile_estimation(1)  # grace consumed
    _write_epoch(cb, 2, _failing_epoch())

    with pytest.raises(RuntimeError, match="grace epoch did not recover"):
        cb._reconcile_estimation(2)


def test_reconcile_grace_rearms_after_recovery(thinking_tokenizer, tmp_path):
    # Exceed at epoch 1, recover at epoch 2 -> epoch 3 may exceed again without aborting.
    cb = _callback(thinking_tokenizer, tmp_path, max_failed=0.1)
    _write_epoch(cb, 0, [0.5] * 10)
    cb._reconcile_estimation(0)
    _write_epoch(cb, 1, _failing_epoch())
    cb._reconcile_estimation(1)
    _write_epoch(cb, 2, [0.7] * 10)
    cb._reconcile_estimation(2)
    _write_epoch(cb, 3, _failing_epoch())

    cb._reconcile_estimation(3)  # no raise: epoch 2 recovered, so the grace is re-armed

    assert _read_stats(cb, 2)["exceeded_threshold"] is False
    assert _read_stats(cb, 3)["exceeded_threshold"] is True


def test_reconcile_missing_prev_stats_grants_grace(thinking_tokenizer, tmp_path):
    # Epoch 0's parquet exists (so backfill works) but its stats marker does not -- a pre-feature
    # dump or a wiped data dir. Be permissive rather than fatal.
    cb = _callback(thinking_tokenizer, tmp_path, max_failed=0.1)
    _write_epoch(cb, 0, [0.5] * 10)
    _write_epoch(cb, 1, _failing_epoch())
    assert not cb._stats_path_for_epoch(0).exists()

    cb._reconcile_estimation(1)  # no raise

    assert _read_stats(cb, 1)["exceeded_threshold"] is True


def test_grace_epoch_aborts_when_no_rows_selected(thinking_tokenizer, tmp_path):
    # Every row's entropy equals the teacher's -> zero gain -> the sampler drops all of them. A grace
    # epoch must not hand training an empty sample; fail here rather than deep inside create_sample.
    train = _train_dataset(thinking_tokenizer, [("t", 100)])
    cb = _callback(thinking_tokenizer, tmp_path, train_dataset=train)
    _write_epoch(cb, 0, [0.2] * 10, teacher=0.2)
    cb._reconcile_estimation(0)  # 0% failed: not a grace epoch, so no guard despite 0 selected
    assert _read_stats(cb, 0)["selected_samples"] == 0
    path1 = _write_epoch(cb, 1, _failing_epoch(value=0.2), teacher=0.2)

    with pytest.raises(RuntimeError, match="no rows with a positive score"):
        cb._reconcile_estimation(1)

    # The parquet keeps its unbackfilled NaNs, so a resume re-derives the same high failure rate and
    # aborts again rather than sailing past the guard into an opaque create_sample error.
    assert pd.read_parquet(path1)["entropy_value"].isna().sum() == 3
    with pytest.raises(RuntimeError, match="no rows with a positive score"):
        cb._reconcile_estimation(1)


def test_reconcile_epoch0_keeps_residual_nan(thinking_tokenizer, tmp_path):
    # Epoch 0 has no previous epoch to backfill from; residual NaN stays (sampler drops it).
    cb = _callback(thinking_tokenizer, tmp_path)
    entropy = [0.5] * 20
    entropy[5] = nan  # 5% <= 10%, no abort
    path0 = _write_epoch(cb, 0, entropy)

    cb._reconcile_estimation(0)

    out = pd.read_parquet(path0)
    assert out["entropy_value"].isna().sum() == 1


def test_reconcile_records_selected_samples_per_dataset(thinking_tokenizer, tmp_path):
    # Two train adapters with different top_k score the same parquet; all 20 rows are positive
    # (entropy 0.9 > teacher 0.2), so each selects min(top_k, 20).
    train = _train_dataset(thinking_tokenizer, [("reasoning", 100), ("single", 50)])
    cb = _callback(thinking_tokenizer, tmp_path, train_dataset=train)
    _write_epoch(cb, 0, [0.9] * 20, teacher=0.2)

    cb._reconcile_estimation(0)

    stats = _read_stats(cb, 0)
    assert stats["selected_samples"] == 40
    assert stats["selected_samples_by_dataset"] == {"reasoning": 20, "single": 20}


def test_reconcile_selected_samples_excludes_non_positive(thinking_tokenizer, tmp_path):
    train = _train_dataset(thinking_tokenizer, [("t", 100)])
    cb = _callback(thinking_tokenizer, tmp_path, train_dataset=train)
    # 12 positive (gain 0.7), 8 non-positive (entropy == teacher -> gain 0).
    _write_epoch(cb, 0, [0.9] * 12 + [0.2] * 8, teacher=0.2)

    cb._reconcile_estimation(0)

    assert _read_stats(cb, 0)["selected_samples"] == 12


def test_reconcile_selected_samples_uses_post_backfill_values(thinking_tokenizer, tmp_path):
    # Row 3 fails this epoch (NaN) and is backfilled to a positive-gain value from epoch 0.
    # The count must reflect the post-backfill df (20), not the pre-backfill measured rows (19).
    train = _train_dataset(thinking_tokenizer, [("t", 100)])
    cb = _callback(thinking_tokenizer, tmp_path, train_dataset=train)
    n = 20
    _write_epoch(cb, 0, [0.9] * n, teacher=0.2)
    epoch1 = [0.9] * n
    epoch1[3] = nan  # 5% <= 10%, no abort
    _write_epoch(cb, 1, epoch1, teacher=0.2)

    cb._reconcile_estimation(1)

    assert _read_stats(cb, 1)["selected_samples"] == 20


def test_reconcile_without_train_dataset_records_none(thinking_tokenizer, tmp_path):
    cb = _callback(thinking_tokenizer, tmp_path)  # no train_dataset
    _write_epoch(cb, 0, [0.5] * 20, teacher=0.2)

    cb._reconcile_estimation(0)

    stats = _read_stats(cb, 0)
    assert stats["selected_samples"] is None
    assert stats["selected_samples_by_dataset"] is None


def test_aggregate_carries_selected_samples_and_tolerates_missing(tmp_path):
    # Roll-up across a new-format stats file (has selected_samples) and a pre-feature one (missing it).
    base = tmp_path / "resampling_trainer_data"
    d0 = base / "0" / "complexity_estimation"
    d0.mkdir(parents=True)
    (d0 / "estimation_stats.json").write_text(
        json.dumps(
            {
                "epoch": 0,
                "total_rows": 100,
                "failed_rows": 0,
                "failure_rate": 0.0,
                "backfilled": 0,
                "residual_failed": 0,
                "selected_samples": 40,
                "selected_samples_by_dataset": {"t": 40},
            }
        )
    )
    d1 = base / "1" / "complexity_estimation"
    d1.mkdir(parents=True)
    (d1 / "estimation_stats.json").write_text(
        json.dumps(
            {
                "epoch": 1,
                "total_rows": 100,
                "failed_rows": 2,
                "failure_rate": 0.02,
                "backfilled": 2,
                "residual_failed": 0,
            }
        )
    )

    trainer = ResamplingTrainer.__new__(ResamplingTrainer)
    trainer.config = SimpleNamespace(  # type: ignore[assignment]
        out_path=str(tmp_path),
        complexity_evaluation_dataset=SimpleNamespace(dataset=SimpleNamespace(dataset_id="mmlu_teacher_entropy")),
    )

    trainer._aggregate_estimation_stats()

    summary = json.loads((base / "complexity_estimation_failures.json").read_text())
    assert summary["selected_samples_by_epoch"] == {"0": 40, "1": None}
    assert summary["grace_epochs"] == []  # neither stats file carries exceeded_threshold
