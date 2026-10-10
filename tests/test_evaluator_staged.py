"""Stage-loop orchestration tests for Evaluator._evaluate_single.

These exercise the outer length-stage loop, cross-sub-chunk pooling of the long
tail, dataset-order assembly, accuracy accounting, and crash-resume — all with a
fake generator + fake dataset, so they are CPU-only and need no model.

The generator's carry/re-prefill correctness is covered by
test_phased_batch_generator.py::TestLengthStageCarry; here we only drive the
control flow above generate().
"""

import types

import pytest

from core.evaluation import evaluator as evaluator_mod
from core.evaluation.evaluator import Evaluator
from core.evaluation.phased_batch_generator import GenerationResult

MAX_NEW = 20
STAGE = 5

# Per-row "true" generation length, keyed by the synthetic prompt token (== row idx).
# Mix of short (finish stage 0), medium (finish stage 1), and full-length (carried
# through every stage, truncated at the final boundary).
TRUE_LEN = {0: 3, 1: 20, 2: 8, 3: 3, 4: 20, 5: 8}


class _FakeGen:
    """Deterministic stand-in for BatchGenerator. Each prompt is a single token == its
    row idx; output length is TRUE_LEN[idx]. Honors carry_at_new_tokens like the real
    one (carry when target exceeds the cap, else finish/truncate)."""

    _PHASE_STEP = 512  # mirrors BatchGenerator (evaluator reads it for an alignment warning)
    crash_on_call: int | None = None
    call_count: int = 0

    def __init__(self, **kwargs):
        self.max_new_tokens = kwargs["max_new_tokens"]

    def generate(
        self,
        prompts,
        checkpoint_path=None,
        carry_at_new_tokens=None,
        resume_generated=None,
        resume_thinking_exhausted=None,
    ):
        _FakeGen.call_count += 1
        if _FakeGen.crash_on_call is not None and _FakeGen.call_count == _FakeGen.crash_on_call:
            raise RuntimeError("simulated GPU crash")

        seqs, unfinished, truncated, tbe = [], [], [], []
        for p in prompts:
            target = TRUE_LEN[p[0]]
            if carry_at_new_tokens is not None and target > carry_at_new_tokens:
                seqs.append(list(range(carry_at_new_tokens)))
                unfinished.append(True)
                truncated.append(False)
            else:
                seqs.append(list(range(target)))
                unfinished.append(False)
                truncated.append(target >= self.max_new_tokens)
            tbe.append(False)
        return GenerationResult(
            sequences=seqs,
            num_truncated=sum(truncated),
            total=len(prompts),
            truncated=truncated,
            thinking_budget_exhausted=tbe,
            unfinished=unfinished,
        )


class _FakeTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return ",".join(str(i) for i in ids)


class _FakeQADataset:
    dataset_id = "fakeds"

    def verify_assistant_response(self, row, response):
        # Deterministic: even ds_index is "correct".
        return response, (row["ds_index"] % 2 == 0)


class _FakeAdapter:
    def __init__(self):
        self.dataset = _FakeQADataset()
        # Each row: a single-token prompt (== idx), a stable row_id, and ds_index so
        # the fake verifier can decide correctness.
        self._rows = [
            {"input_ids": [i], "row_id": f"r{i}", "ds_index": i} for i in range(len(TRUE_LEN))
        ]

    def process_dataset(self):
        return self._rows


def _make_evaluator(tmp_path):
    gen_cfg = types.SimpleNamespace(
        max_new_tokens=MAX_NEW,
        max_thinking_tokens=None,
        max_batch_size=4,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        kv_cache_offload_threshold_gb=200.0,
        kv_cache_spill_dir=None,
        kv_cache_pin_memory=True,
        stage_new_tokens=STAGE,
    )
    config = types.SimpleNamespace(model_path="fake-model", out_path=str(tmp_path), generation=gen_cfg)
    return Evaluator(config, tokenizer=_FakeTokenizer())


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setattr(evaluator_mod, "BatchGenerator", _FakeGen)
    monkeypatch.setattr(evaluator_mod, "CHUNK_SIZE", 2)  # small chunks → real cross-chunk pooling
    _FakeGen.crash_on_call = None
    _FakeGen.call_count = 0
    yield


def test_staged_pooling_order_and_accuracy(tmp_path):
    ev = _make_evaluator(tmp_path)
    result = ev._evaluate_single(_FakeAdapter(), model=None, tokenizer=_FakeTokenizer())

    assert result.total == 6
    # is_correct for even ds_index (0,2,4) → 3/6.
    assert result.correct == 3
    assert result.accuracy == pytest.approx(0.5)
    # Full-length rows (idx 1, 4) hit max_new without EOS → truncated.
    assert result.num_truncated == 2


def test_responses_in_dataset_order(tmp_path):
    ev = _make_evaluator(tmp_path)
    ev._evaluate_single(_FakeAdapter(), model=None, tokenizer=_FakeTokenizer())

    import pandas as pd

    df = pd.read_parquet(tmp_path / "fakeds" / "responses.parquet")
    assert list(df["row_id"]) == [f"r{i}" for i in range(6)]
    # ds_index helper key must not leak into the saved schema.
    assert "ds_index" not in df.columns


def test_resume_after_crash_matches_clean_run(tmp_path):
    # Clean reference run in its own out dir.
    ref = _make_evaluator(tmp_path / "clean")._evaluate_single(_FakeAdapter(), model=None, tokenizer=_FakeTokenizer())

    # Crash partway through stage 0 (on the 2nd sub-chunk), in a fresh out dir.
    crash_dir = tmp_path / "crash"
    _FakeGen.crash_on_call = 2
    _FakeGen.call_count = 0
    with pytest.raises(RuntimeError, match="simulated GPU crash"):
        _make_evaluator(crash_dir)._evaluate_single(_FakeAdapter(), model=None, tokenizer=_FakeTokenizer())

    # Relaunch: replay from stage 0, cache-hit the completed sub-chunk, finish the rest.
    _FakeGen.crash_on_call = None
    _FakeGen.call_count = 0
    resumed = _make_evaluator(crash_dir)._evaluate_single(_FakeAdapter(), model=None, tokenizer=_FakeTokenizer())

    assert (resumed.total, resumed.correct, resumed.num_truncated) == (ref.total, ref.correct, ref.num_truncated)
    assert resumed.accuracy == pytest.approx(ref.accuracy)
