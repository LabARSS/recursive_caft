"""On resume, a dumped per-epoch complexity may only be reused when that epoch's weights were
checkpointed (epoch in save_schedule) — otherwise the epoch is re-trained from an earlier checkpoint
and its dump is stale, so it must be recomputed on the current weights.
"""

import types

import pytest

from core.training.resampling_trainer import EstimateComplexityCallback


def _callback(out_path, save_schedule):
    # The reuse-gating predicates only touch out_path / dataset_id / save_schedule and the marker
    # file, so a lightweight stand-in for the dataset adapter is enough (no tokenizer/model needed).
    return EstimateComplexityCallback(
        complexity_evaluation_dataset=types.SimpleNamespace(
            dataset=types.SimpleNamespace(dataset_id="mmlu_teacher_entropy")
        ),
        complexity_estimator=None,
        complexity_estimation_runner_generation_config=None,
        out_path=out_path,
        save_schedule=save_schedule,
    )


def _write_marker(cb, epoch):
    p = cb._stats_path_for_epoch(epoch)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}")


def test_should_reuse_only_for_checkpointed_epochs(tmp_path):
    cb = _callback(tmp_path, save_schedule=[1, 3])
    for epoch in (1, 2, 3):
        _write_marker(cb, epoch)

    assert cb._should_reuse(1) is True  # checkpointed + marker
    assert cb._should_reuse(2) is False  # not checkpointed -> recompute even though marked
    assert cb._should_reuse(3) is True
    assert cb._should_reuse(4) is False  # not checkpointed and no marker


def test_should_reuse_requires_a_marker(tmp_path):
    cb = _callback(tmp_path, save_schedule=[1, 3])
    # Epoch 3 is checkpointed but its complexity was not finalized yet.
    assert cb._should_reuse(3) is False


def test_no_schedule_is_permissive(tmp_path):
    # save_schedule=None ⇒ HF checkpoints every epoch, so any finalized dump is reusable.
    cb = _callback(tmp_path, save_schedule=None)
    _write_marker(cb, 2)
    assert cb._is_checkpoint_epoch(2) is True
    assert cb._should_reuse(2) is True


def test_on_epoch_begin_reuses_checkpointed_marked_epoch(tmp_path):
    cb = _callback(tmp_path, save_schedule=[1, 3])
    _write_marker(cb, 1)
    # Checkpointed + marked -> short-circuits without ever touching kwargs["model"].
    assert cb.on_epoch_begin(None, types.SimpleNamespace(epoch=1.0), None) is None


def test_on_epoch_begin_recomputes_non_checkpointed_epoch(tmp_path):
    cb = _callback(tmp_path, save_schedule=[1, 3])
    _write_marker(cb, 2)  # epoch 2 has a dump, but it is NOT a checkpoint epoch
    # Not reusable -> falls through to estimate(), which reads kwargs["model"] (absent here),
    # proving it did not reuse the stale dump.
    with pytest.raises(KeyError):
        cb.on_epoch_begin(None, types.SimpleNamespace(epoch=2.0), None)
