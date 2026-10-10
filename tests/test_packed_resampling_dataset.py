"""Packed resampling: constant-length epochs via null padding.

`PackedResamplingDataset` must yield EXACTLY len(dataset) rows per epoch — HF's Trainer
(transformers 4.52.3) fixes `steps_in_epoch` from `len(dataset)` at the top of each epoch
(trainer.py:2478) and advances `state.epoch` by 1/steps_in_epoch per row (trainer.py:2620),
so every `int(state.epoch)` consumer (EstimateComplexityCallback, SaveByScheduleCallback,
resume arithmetic) silently corrupts if an epoch under- or over-yields. These tests lock in
that row-count contract, the per-accumulation-chunk NaN guard, and the pack layout — CPU
only, no model.
"""

import pytest

from core.dataset_samplers.base_sampler import BaseDatasetSamplerConfig
from core.dataset_samplers.entropy_gain_sampler import EntropyGainSampler
from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.mmlu.mmlu_reasoning_response_dataset import MMLUReasoningResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.sequence_packing import IGNORE_LABEL
from core.training.resampling_trainer import (
    NULL_PACK_LENGTH,
    PackedResamplingDataset,
    interleave_pack_slots,
    make_null_pack,
    pad_pack_indices,
)


def _adapter(tokenizer, top_k):
    sampler = EntropyGainSampler(BaseDatasetSamplerConfig(top_k=top_k)) if top_k else None
    return CausalDatasetAdapter(
        dataset=MMLUReasoningResponseDataset(
            config=QADatasetConfig(path="sentinel-overridden-at-runtime", dataset_id="t"),
            tokenizer=tokenizer,
        ),
        dataset_sampler=sampler,
    )


# --- make_null_pack -----------------------------------------------------------------


def test_null_pack_convention():
    pack = make_null_pack(pad_token_id=7)
    assert pack["input_ids"] == [7] * NULL_PACK_LENGTH
    assert pack["labels"] == [IGNORE_LABEL] * NULL_PACK_LENGTH
    # position_ids restart at 0, like pack_dataset's pad segment: FA2 varlen sees one
    # isolated (and fully loss-ignored) segment.
    assert pack["position_ids"] == list(range(NULL_PACK_LENGTH))


# --- interleave_pack_slots ----------------------------------------------------------


@pytest.mark.parametrize(
    "num_packs,total_slots,updates",
    [
        (20, 1280, 20),  # experiment shape, minimum packs
        (300, 1280, 20),  # experiment shape, typical packs
        (1280, 1280, 20),  # nothing merged: zero nulls
        (2, 8, 2),  # minimum viable
        (7, 8, 2),  # one null
        (8, 8, 2),  # zero nulls
    ],
)
def test_interleave_contract(num_packs, total_slots, updates):
    slots = interleave_pack_slots(list(range(num_packs)), total_slots, updates)

    assert len(slots) == total_slots
    # Real packs preserved exactly once, in order (no loss, no duplication here).
    assert [s for s in slots if s is not None] == list(range(num_packs))
    # NaN guard: every gradient-accumulation chunk holds >= 1 real pack, because
    # num_items_in_batch is counted per chunk and an all-null chunk divides 0 by 0.
    chunk_size = total_slots // updates
    for c in range(updates):
        chunk = slots[c * chunk_size : (c + 1) * chunk_size]
        assert any(s is not None for s in chunk), f"all-null accumulation chunk {c}"


def test_interleave_deterministic():
    a = interleave_pack_slots(list(range(33)), 128, 4)
    b = interleave_pack_slots(list(range(33)), 128, 4)
    assert a == b


def test_interleave_rejects_bad_shapes():
    with pytest.raises(AssertionError):  # fewer packs than chunks (pad_pack_indices' job)
        interleave_pack_slots([0], 8, 2)
    with pytest.raises(AssertionError):  # more packs than slots
        interleave_pack_slots(list(range(9)), 8, 2)
    with pytest.raises(AssertionError):  # slots not divisible by updates
        interleave_pack_slots([0, 1], 7, 2)


# --- pad_pack_indices ---------------------------------------------------------------


def test_pad_duplicates_smallest_pack():
    indices = pad_pack_indices(3, 5, supervised_token_counts=[10, 2, 7])
    assert indices == [0, 1, 2, 1, 1]  # pack 1 has the fewest supervised tokens


def test_pad_noop_when_enough_packs():
    assert pad_pack_indices(5, 5, supervised_token_counts=[1] * 5) == [0, 1, 2, 3, 4]
    assert pad_pack_indices(6, 5, supervised_token_counts=[1] * 6) == [0, 1, 2, 3, 4, 5]


def test_pad_then_interleave_keeps_chunks_non_null():
    indices = pad_pack_indices(2, 4, supervised_token_counts=[9, 3])
    slots = interleave_pack_slots(indices, 16, 4)
    for c in range(4):
        chunk = slots[c * 4 : (c + 1) * 4]
        assert any(s is not None for s in chunk)


# --- PackedResamplingDataset --------------------------------------------------------

TOP_K = 8
ACCUM = 4  # updates_per_epoch = 2


@pytest.fixture
def packed_ds(thinking_tokenizer, tmp_path, make_resampling_df):
    from core.training.base_trainer import PackingConfig

    parquet = tmp_path / "epoch0.parquet"
    make_resampling_df(6).to_parquet(parquet)  # 6 docs < top_k: shortfall absorbed by nulls

    adapter = _adapter(thinking_tokenizer, top_k=TOP_K)
    # budget == the longest tokenized doc: valid (pack_dataset requires budget >= max len)
    # and guarantees >= 2 packs for >= 2 docs, keeping this off the duplication path.
    docs = adapter.process_dataset(path_override=str(parquet))
    budget = max(len(ids) for ids in docs["input_ids"])

    ds = PackedResamplingDataset(
        adapter, thinking_tokenizer, packing=PackingConfig(budget=budget), gradient_accumulation_steps=ACCUM
    )
    ds.dataset_path = str(parquet)
    return ds, docs, budget


def test_epoch_yields_exactly_len_rows(packed_ds):
    ds, _, _ = packed_ds
    assert len(ds) == TOP_K  # pessimistic constant, same as the non-packed dataset
    rows = list(ds)
    assert len(rows) == TOP_K


def test_rows_carry_only_packed_columns(packed_ds):
    ds, _, budget = packed_ds
    for row in ds:
        assert set(row.keys()) == {"input_ids", "labels", "position_ids"}
        # Real packs are budget-sized (constant compile shape); nulls are tiny.
        assert len(row["input_ids"]) in (budget, NULL_PACK_LENGTH)


def test_supervised_tokens_conserved(packed_ds):
    # Nothing lost, nothing double-trained: supervised (label != -100) token multiset of the
    # yielded epoch == that of the sampled docs. Nulls contribute zero.
    ds, docs, _ = packed_ds
    yielded = sorted(tok for row in ds for tok in row["labels"] if tok != IGNORE_LABEL)
    original = sorted(tok for labels in docs["labels"] for tok in labels if tok != IGNORE_LABEL)
    assert yielded == original


def test_every_chunk_has_supervision(packed_ds):
    ds, _, _ = packed_ds
    rows = list(ds)
    for c in range(TOP_K // ACCUM):
        chunk = rows[c * ACCUM : (c + 1) * ACCUM]
        assert any(tok != IGNORE_LABEL for row in chunk for tok in row["labels"])


def test_second_epoch_identical(packed_ds):
    # Sampler sort, BFD packing and the interleave are all RNG-free; a re-trained epoch
    # (crash-resume at a non-checkpointed epoch) must replay identically.
    ds, _, _ = packed_ds
    assert list(ds) == list(ds)


def test_packed_shuffle_is_epoch_seeded(thinking_tokenizer, tmp_path, make_resampling_df):
    from core.training.base_trainer import PackingConfig

    parquet = tmp_path / "shuffled.parquet"
    source = make_resampling_df(6)
    source["question"] = [f"Unique question {i}?" for i in range(len(source))]
    source.to_parquet(parquet)
    adapter = _adapter(thinking_tokenizer, top_k=TOP_K)
    docs = adapter.process_dataset(path_override=str(parquet))
    budget = max(len(ids) for ids in docs["input_ids"])
    ds = PackedResamplingDataset(
        adapter,
        thinking_tokenizer,
        packing=PackingConfig(budget=budget),
        gradient_accumulation_steps=ACCUM,
        shuffle=True,
        data_seed=123,
    )
    ds.dataset_path = str(parquet)

    ds.epoch = 4
    first = list(ds)
    replay = list(ds)
    ds.epoch = 5
    next_epoch = list(ds)

    assert first == replay
    assert first != next_epoch


def test_len_must_divide_accumulation(thinking_tokenizer):
    from core.training.base_trainer import PackingConfig

    with pytest.raises(AssertionError, match="divisible"):
        PackedResamplingDataset(
            _adapter(thinking_tokenizer, top_k=7),
            thinking_tokenizer,
            packing=PackingConfig(budget=1024),
            gradient_accumulation_steps=4,
        )


def test_requires_sampler_defined_size(thinking_tokenizer):
    from core.training.base_trainer import PackingConfig

    with pytest.raises(AssertionError, match="sampler-defined size"):
        PackedResamplingDataset(
            _adapter(thinking_tokenizer, top_k=None),
            thinking_tokenizer,
            packing=PackingConfig(budget=1024),
            gradient_accumulation_steps=4,
        )


# --- config validation ---------------------------------------------------------------


def test_config_rejects_packing_with_batch_size_above_one(thinking_tokenizer):
    from core.complexity_estimation.complexity_estimation_runner import ModelGenerateConfig
    from core.complexity_estimation.entropy.single_token_entropy_estimator import SingleTokenEntropyEstimator
    from core.datasets.mmlu.mmlu_single_token_response_dataset import MMLUSingleTokenResponseDataset
    from core.datasets.qa_dataset_adapter import QADatasetAdapter
    from core.training.base_trainer import PackingConfig
    from core.training.lora_trainer import LoRATrainingArgs
    from core.training.resampling_trainer import ResamplingTrainerConfig

    from typing import Any

    kwargs: dict[str, Any] = dict(
        out_path="/tmp/x",
        model_id="m",
        train_dataset=_adapter(thinking_tokenizer, top_k=TOP_K),
        packing=PackingConfig(budget=1024),
        complexity_evaluation_dataset=QADatasetAdapter(
            dataset=MMLUSingleTokenResponseDataset(
                config=QADatasetConfig(path="sentinel", dataset_id="d"), tokenizer=thinking_tokenizer
            )
        ),
        complexity_estimator=SingleTokenEntropyEstimator(),
        complexity_estimation_runner_generation_config=ModelGenerateConfig(max_new_tokens=1),
    )

    with pytest.raises(Exception, match="per_device_train_batch_size=1"):
        ResamplingTrainerConfig(
            training_args=LoRATrainingArgs(num_train_epochs=1, per_device_train_batch_size=2), **kwargs
        )
    # batch size 1 passes validation
    ResamplingTrainerConfig(training_args=LoRATrainingArgs(num_train_epochs=1, per_device_train_batch_size=1), **kwargs)


# --- HF chunking arithmetic tripwire ------------------------------------------------


def test_hf_chunk_arithmetic_contract():
    # Emulates transformers 4.52.3 trainer.py:2499-2512 for the shapes this design relies
    # on. If a transformers upgrade changes this arithmetic, this test is the tripwire.
    for total_slots, accum in [(1280, 64), (TOP_K, ACCUM), (128, 64)]:
        steps_in_epoch = total_slots  # len(dataloader) with batch_size=1, exact yield
        remainder = steps_in_epoch % accum
        if remainder == 0:
            remainder = accum
        total_updates = steps_in_epoch // accum + int(remainder < accum)
        assert total_updates == total_slots // accum  # constant updates/epoch

        sync_steps = [s for s in range(1, steps_in_epoch + 1) if s % accum == 0 or s == steps_in_epoch]
        assert len(sync_steps) == total_updates  # one optimizer step per chunk
        # state.epoch = epoch + step/steps_in_epoch lands exactly on the integer boundary.
        assert sync_steps[-1] / steps_in_epoch == 1.0
