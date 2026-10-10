"""EstimateComplexityCallback subprocess routing + escape hatch + spec picklability.

CPU-only — no model, no GPU. The supervised default path is exercised by spying on
_estimate_in_subprocess; the COMPLEXITY_SUPERVISE=0 escape hatch is exercised by spying on
ComplexityEstimationRunner.estimate. The GPU end-to-end (real save_pretrained + spawn +
reload) is verified manually.
"""

import pickle
from types import SimpleNamespace

import pytest

import core.training.resampling_trainer as rt
from core.complexity_estimation.complexity_estimation_runner import (
    ComplexityEstimationRunner,
    ComplexityEstimationRunnerConfig,
    ModelGenerateConfig,
)
from core.complexity_estimation.entropy.single_token_entropy_estimator import SingleTokenEntropyEstimator
from core.datasets.mmlu.mmlu_single_token_response_dataset import MMLUSingleTokenResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.training.resampling_trainer import EstimateComplexityCallback


def _adapter(thinking_tokenizer):
    return QADatasetAdapter(
        dataset=MMLUSingleTokenResponseDataset(
            config=QADatasetConfig(path="sentinel", dataset_id="mmlu_teacher_entropy"),
            tokenizer=thinking_tokenizer,
        )
    )


def _callback(thinking_tokenizer, out_path):
    return EstimateComplexityCallback(
        complexity_evaluation_dataset=_adapter(thinking_tokenizer),
        complexity_estimator=SingleTokenEntropyEstimator(),
        complexity_estimation_runner_generation_config=ModelGenerateConfig(max_new_tokens=1),
        out_path=out_path,
    )


def test_supervise_off_runs_in_process_without_spawning(thinking_tokenizer, tmp_path, monkeypatch):
    monkeypatch.setenv("COMPLEXITY_SUPERVISE", "0")
    cb = _callback(thinking_tokenizer, tmp_path)

    captured = {}

    def fake_estimate(self, dataset_adapter, model):
        captured["model"] = model
        captured["adapter"] = dataset_adapter

    monkeypatch.setattr(ComplexityEstimationRunner, "estimate", fake_estimate)
    monkeypatch.setattr(EstimateComplexityCallback, "_reconcile_estimation", lambda self, epoch: None)
    # If it tried to spawn, this would blow up.
    monkeypatch.setattr(rt, "supervise_unit", lambda *a, **k: pytest.fail("must not spawn when SUPERVISE=0"))

    sentinel = object()
    cb.on_epoch_begin(None, SimpleNamespace(epoch=0.0), None, model=sentinel)

    assert captured["model"] is sentinel
    assert captured["adapter"] is cb._complexity_evaluation_dataset


def test_default_routes_to_subprocess(thinking_tokenizer, tmp_path, monkeypatch):
    monkeypatch.delenv("COMPLEXITY_SUPERVISE", raising=False)
    cb = _callback(thinking_tokenizer, tmp_path)

    calls = {}

    def fake_subprocess(self, epoch, model, runner_config):
        calls["epoch"] = epoch
        calls["model"] = model
        calls["out_path"] = runner_config.out_path

    monkeypatch.setattr(EstimateComplexityCallback, "_estimate_in_subprocess", fake_subprocess)
    monkeypatch.setattr(EstimateComplexityCallback, "_reconcile_estimation", lambda self, epoch: None)
    monkeypatch.setattr(
        ComplexityEstimationRunner, "estimate", lambda *a, **k: pytest.fail("default path must not run in-process")
    )

    sentinel = object()
    cb.on_epoch_begin(None, SimpleNamespace(epoch=0.0), None, model=sentinel)

    assert calls["epoch"] == 0
    assert calls["model"] is sentinel
    assert calls["out_path"] == cb.out_path_for_epoch(0).as_posix()


def test_missing_model_raises_before_any_spawn(thinking_tokenizer, tmp_path, monkeypatch):
    # The recompute branch must read kwargs["model"] first; with no model it raises KeyError before
    # touching the filesystem or spawning. (Locks the ordering the supervised path depends on.)
    monkeypatch.delenv("COMPLEXITY_SUPERVISE", raising=False)
    cb = _callback(thinking_tokenizer, tmp_path)
    monkeypatch.setattr(
        EstimateComplexityCallback, "_estimate_in_subprocess", lambda *a, **k: pytest.fail("spawned without a model")
    )
    with pytest.raises(KeyError):
        cb.on_epoch_begin(None, SimpleNamespace(epoch=0.0), None)


def test_spec_pickles_and_preserves_thinking_attrs(thinking_tokenizer, tmp_path):
    # The exact 5-tuple the callback ships to the worker must round-trip, including the tokenizer's
    # dynamically-attached thinking-token ids carried inside the dataset adapter.
    adapter = _adapter(thinking_tokenizer)
    runner_config = ComplexityEstimationRunnerConfig(
        out_path=(tmp_path / "out.parquet").as_posix(),
        answer_field_name="estimation_phase_answer",
        answer_correctness_field_name="estimation_phase_answer_correctness",
        generate_config=ModelGenerateConfig(max_new_tokens=1),
    )
    spec = (tmp_path.as_posix(), runner_config, SingleTokenEntropyEstimator(), adapter, "flash_attention_2")

    model_dir, cfg2, est2, adapter2, attn2 = pickle.loads(pickle.dumps(spec, protocol=pickle.HIGHEST_PROTOCOL))

    assert model_dir == tmp_path.as_posix()
    assert cfg2.out_path == runner_config.out_path
    assert isinstance(est2, SingleTokenEntropyEstimator)
    assert attn2 == "flash_attention_2"
    tok = adapter2.dataset.tokenizer
    assert tok.thinking_start_token_id == thinking_tokenizer.thinking_start_token_id
    assert tok.thinking_end_token_id == thinking_tokenizer.thinking_end_token_id
