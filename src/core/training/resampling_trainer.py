import gc
import json
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path
from typing import override

import pandas as pd
import torch
from pydantic import model_validator
from torch.utils.data import IterableDataset
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from core.complexity_estimation.complexity_estimation_runner import (
    BaseComplexityEstimator,
    ComplexityEstimationRunner,
    ComplexityEstimationRunnerConfig,
    ModelGenerateConfig,
    QADatasetAdapter,
)
from core.datasets.abstract_dataset_adapter import AbstractDatasetAdapter
from core.datasets.sequence_packing import IGNORE_LABEL, pack_dataset, sequence_length_stats
from core.training.base_trainer import PackingConfig
from core.training.callbacks.save_thinking_token_rows import write_thinking_token_rows
from core.training.lora_trainer import LoRATrainer, LoRATrainerConfig
from core.utils.logger import logger
from core.utils.subprocess_supervision import supervise_unit

# Standalone worker that runs one epoch's estimate() in a fresh, supervised process.
_COMPLEXITY_WORKER = str(Path(__file__).with_name("_complexity_worker.py"))


class ResamplingDataset(IterableDataset):
    def __init__(
        self,
        dataset: AbstractDatasetAdapter,
        tokenizer: PreTrainedTokenizer,
        shuffle: bool = False,
        data_seed: int = 42,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.data_seed = data_seed
        self.epoch = 0
        self.selection_epoch = 0
        self._logged_samples = False

        self.dataset_path: str | None = None

    def __len__(self) -> int:
        # HF Trainer requires the train dataset to be sized (otherwise it raises and demands
        # max_steps). Each adapter reports its post-sampling size (the sampler's top_k, summed
        # across a MergedDatasetAdapter). This must stay O(1) / I/O-free: __len__ is evaluated in
        # Trainer.__init__ while dataset_path is still None, before the first on_epoch_begin sets it.
        size = self.dataset.sampled_size()
        if size is not None:
            return size
        return len(self.dataset.process_dataset(path_override=self.dataset_path))

    def __iter__(self):
        # Calling process_dataset to re-sample dataset after EstimateComplexityCallback runs
        self.dataset.set_epoch(self.selection_epoch)
        dataset = self.dataset.process_dataset(
            path_override=self.dataset_path,
            shuffle=self.shuffle,
            shuffle_seed=self.data_seed + self.epoch,
        )

        if not self._logged_samples and len(dataset) > 0:
            first_sample = dataset[0]
            logger.info("Dataset samples")
            logger.info("Train")
            logger.info(f"Input: {self.tokenizer.decode(first_sample['input_ids'])}")
            labels = [tok for tok in first_sample["labels"] if tok != -100]
            logger.info(f"Labels: {self.tokenizer.decode(labels)}")
            self._logged_samples = True

        yield from dataset


# Null packs are tiny, NOT budget-sized: their forward cost must be negligible (an epoch can
# hold ~1000 of them). 8 matches the pad_to_multiple_of=8 convention elsewhere; torch.compile
# just sees one extra static shape besides `budget`.
NULL_PACK_LENGTH = 8


def make_null_pack(pad_token_id: int, length: int = NULL_PACK_LENGTH) -> dict:
    """An all-ignored block: pad input_ids, IGNORE_LABEL labels, position_ids restarting at 0
    (the same convention pack_dataset uses for a block's pad segment, so FA2 varlen treats it
    as one isolated, loss-ignored segment)."""
    return {
        "input_ids": [pad_token_id] * length,
        "labels": [IGNORE_LABEL] * length,
        "position_ids": list(range(length)),
    }


def pad_pack_indices(num_packs: int, updates_per_epoch: int, supervised_token_counts: list[int]) -> list[int]:
    """Pack indices for an epoch with fewer packs than accumulation chunks.

    interleave_pack_slots needs >= 1 real pack per chunk (see its NaN rationale), so when the
    sampler selected so few docs that num_packs < updates_per_epoch, the pack with the fewest
    supervised tokens is duplicated up to that minimum. Duplicated packs are trained more than
    once that epoch -- bounded, rare (late-run selection collapse), and loudly logged."""
    assert num_packs > 0, "Cannot pad an epoch with zero packs"
    indices = list(range(num_packs))
    if num_packs >= updates_per_epoch:
        return indices

    smallest = min(range(num_packs), key=lambda i: supervised_token_counts[i])
    duplicates = updates_per_epoch - num_packs
    logger.warning(
        f"Resampling packing: only {num_packs} packs for {updates_per_epoch} accumulation chunks; "
        f"duplicating the smallest pack ({supervised_token_counts[smallest]} supervised tokens) "
        f"{duplicates}x so no chunk is all-null."
    )
    return indices + [smallest] * duplicates


def interleave_pack_slots(pack_indices: list[int], total_slots: int, updates_per_epoch: int) -> list[int | None]:
    """Lay out one epoch as exactly total_slots slots: a real-pack index or None (null pack).

    HF's Trainer splits the epoch into updates_per_epoch gradient-accumulation chunks of
    total_slots/updates_per_epoch rows and counts num_items_in_batch (labels != -100) per
    chunk. An all-null chunk would make the model divide a zero loss-sum by zero items ->
    NaN gradients, so every chunk gets its contiguous share of real packs first, then nulls.
    Requires updates_per_epoch <= len(pack_indices) <= total_slots (see pad_pack_indices)."""
    num_packs = len(pack_indices)
    assert total_slots % updates_per_epoch == 0, (
        f"total_slots {total_slots} must be divisible by updates_per_epoch {updates_per_epoch}"
    )
    assert updates_per_epoch <= num_packs <= total_slots, (
        f"Need updates_per_epoch {updates_per_epoch} <= num_packs {num_packs} <= total_slots {total_slots}"
    )

    chunk_size = total_slots // updates_per_epoch
    slots: list[int | None] = []
    for chunk in range(updates_per_epoch):
        chunk_packs = pack_indices[
            num_packs * chunk // updates_per_epoch : num_packs * (chunk + 1) // updates_per_epoch
        ]
        assert 1 <= len(chunk_packs) <= chunk_size
        slots.extend(chunk_packs)
        slots.extend([None] * (chunk_size - len(chunk_packs)))
    return slots


class PackedResamplingDataset(ResamplingDataset):
    """ResamplingDataset that sequence-packs each epoch's re-sample while keeping the
    pessimistic constant __len__ (the sampler's top_k doc count).

    HF's Trainer fixes steps_in_epoch from len(dataset) at the top of every epoch, BEFORE
    on_epoch_begin sets the epoch's parquet path (transformers 4.52.3 trainer.py:2478->2483,
    iter at 2497), and every resampling callback keys on int(state.epoch), which stays
    integral only if the iterator yields exactly len(dataset) rows. Packing merges docs
    (num_packs <= num_docs <= top_k), so the doc count is a valid constant: each epoch yields
    the real packs padded up to it with tiny "null" packs. Nulls carry zero supervised
    tokens -> exactly zero gradient under the Trainer's token-normalized loss
    (model_accepts_loss_kwargs, asserted at trainer build). Updates/epoch, resume arithmetic
    and the LR schedule stay byte-identical to the non-packed path. Side effect: epochs where
    drop_non_positive selects fewer than top_k docs no longer desync HF's epoch accounting --
    the nulls absorb the shortfall."""

    def __init__(
        self,
        dataset: AbstractDatasetAdapter,
        tokenizer: PreTrainedTokenizer,
        packing: PackingConfig,
        gradient_accumulation_steps: int,
        shuffle: bool = False,
        data_seed: int = 42,
    ):
        super().__init__(dataset, tokenizer, shuffle=shuffle, data_seed=data_seed)
        assert dataset.sampled_size() is not None, (
            "Packed resampling needs a sampler-defined size: __len__ must be a constant "
            "independent of the per-epoch parquet (packs vary per epoch; docs-slots may not)."
        )
        assert len(self) % gradient_accumulation_steps == 0, (
            f"Sampled size {len(self)} must be divisible by gradient_accumulation_steps "
            f"{gradient_accumulation_steps} so accumulation chunks align with the null interleave."
        )
        self.packing = packing
        self.gradient_accumulation_steps = gradient_accumulation_steps

    @override
    def __iter__(self):
        self.dataset.set_epoch(self.selection_epoch)
        dataset = self.dataset.process_dataset(
            path_override=self.dataset_path,
            shuffle=self.shuffle,
            shuffle_seed=self.data_seed + self.epoch,
        )
        assert len(dataset) > 0, "Packed resampling got an empty epoch sample"

        lengths = [len(ids) for ids in dataset["input_ids"]]
        sequence_length_stats(lengths)
        pad_token_id = self.tokenizer.pad_token_id
        assert isinstance(pad_token_id, int), "Tokenizer must have an integer pad_token_id for packing"
        # pack_dataset also enforces budget >= the longest sequence.
        packed = pack_dataset(dataset, budget=self.packing.budget, pad_token_id=pad_token_id)

        if not self._logged_samples:
            first = packed[0]
            logger.info("Dataset samples")
            logger.info("Train (first pack)")
            logger.info(f"Input: {self.tokenizer.decode(first['input_ids'])}")
            labels = [tok for tok in first["labels"] if tok != IGNORE_LABEL]
            logger.info(f"Labels: {self.tokenizer.decode(labels)}")
            self._logged_samples = True

        total_slots = len(self)
        updates_per_epoch = total_slots // self.gradient_accumulation_steps
        if len(packed) < updates_per_epoch:
            counts = [sum(1 for tok in labs if tok != IGNORE_LABEL) for labs in packed["labels"]]
            pack_indices = pad_pack_indices(len(packed), updates_per_epoch, counts)
        else:
            pack_indices = list(range(len(packed)))
        slots = interleave_pack_slots(pack_indices, total_slots, updates_per_epoch)
        logger.info(
            f"Resampling packing: {len(dataset)} docs -> {len(packed)} packs + "
            f"{slots.count(None)} null slots = {total_slots} rows "
            f"({updates_per_epoch} updates x {self.gradient_accumulation_steps} accumulation steps)"
        )

        null_pack = make_null_pack(pad_token_id)
        for slot in slots:
            yield null_pack if slot is None else packed[slot]


class EstimateComplexityCallback(TrainerCallback):
    def __init__(
        self,
        complexity_evaluation_dataset: QADatasetAdapter,
        complexity_estimator: BaseComplexityEstimator,
        complexity_estimation_runner_generation_config: ModelGenerateConfig,
        out_path: Path,
        max_failed_estimation_fraction: float = 0.1,
        save_schedule: list[int] | None = None,
        train_dataset: AbstractDatasetAdapter | None = None,
        new_ids: list[int] | None = None,
        resampling_schedule: list[int] | None = None,
    ) -> None:
        super().__init__()

        self._complexity_evaluation_dataset = complexity_evaluation_dataset
        self._complexity_estimator = complexity_estimator
        self._complexity_estimation_runner_generation_config = complexity_estimation_runner_generation_config
        self._out_path = out_path
        self._max_failed_estimation_fraction = max_failed_estimation_fraction
        self._save_schedule = save_schedule
        # Epochs that actually (re)draw the training sample. None == re-estimate and re-sample every
        # epoch. See ResamplingTrainerConfig.resampling_schedule and resampling_epoch_for.
        self._resampling_schedule = resampling_schedule
        # Used to record, per epoch, how many rows the train sampler(s) actually select from the
        # finalized complexity parquet (after the non-positive-score drop). Optional so the callback
        # can be constructed standalone (e.g. in tests) without the count.
        self._train_dataset = train_dataset
        # The trained <think>/</think> embedding rows live in the base model (not the LoRA adapter),
        # so they must be persisted alongside the snapshot for the estimation subprocess to reload.
        # None when thinking-token embeddings aren't trained. See _estimate_in_subprocess.
        self._new_ids = new_ids

    @override
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        assert state.epoch is not None

        # `epoch` == int(state.epoch) at on_epoch_begin == the number of COMPLETED epochs, i.e. the
        # model state this complexity is measured on — the *previous* training epoch's output. That
        # state is held by the checkpoint whose trainer_state.epoch == epoch, which exists iff
        # `epoch in save_schedule`. (e.g. with schedule [1,3]: dir "1" is model-after-1 == checkpoint
        # epoch=1; dir "2" is model-after-2, which is NOT checkpointed.)
        epoch = int(state.epoch)

        # Frozen epoch: the sample stays whatever the last scheduled epoch drew, and
        # SetResamplingPathCallback keeps the dataset pointed at that epoch's parquet. Nothing to
        # estimate or reconcile -- returning here is what makes a frozen schedule cheap (no weight
        # snapshot, no estimation subprocess). Must precede the kwargs["model"] access below.
        resampling_epoch = self.resampling_epoch_for(epoch)
        if resampling_epoch != epoch:
            logger.info(f"Epoch {epoch} not in resampling_schedule; reusing the epoch {resampling_epoch} sample.")
            return

        # Resume: only reuse a dumped complexity when the weights it was measured on were checkpointed,
        # so resume reloads exactly that model. A non-checkpointed epoch is re-trained from an earlier
        # checkpoint, so its earlier dump is stale for the new trajectory -> recompute it on the
        # current weights (estimate() still resumes from a partial .tmp within that recompute).
        if self._should_reuse(epoch):
            logger.info(f"Reusing dumped complexity for checkpointed epoch {epoch}; skipping re-estimation.")
            return

        if self._is_checkpoint_epoch(epoch) and self._estimation_complete(epoch):
            logger.info(f"Complexity estimate for checkpointed epoch {epoch} already on disk; running reconcile only.")
        else:
            logger.info(f"Estimating complexity for epoch {epoch}...")

            # kwargs["model"] must be the FIRST access on this branch: tests drive on_epoch_begin
            # without a model and rely on the KeyError surfacing here, before any snapshot save or
            # subprocess spawn.
            model: PreTrainedModel = kwargs["model"]

            runner_config = ComplexityEstimationRunnerConfig(
                out_path=self.out_path_for_epoch(epoch).as_posix(),
                answer_field_name="estimation_phase_answer",
                answer_correctness_field_name="estimation_phase_answer_correctness",
                generate_config=self._complexity_estimation_runner_generation_config,
            )

            if os.environ.get("COMPLEXITY_SUPERVISE") == "0":
                # Escape hatch (CPU/local/debug): run in-process, as before — no subprocess.
                ComplexityEstimationRunner(
                    config=runner_config, complexity_estimator=self._complexity_estimator
                ).estimate(dataset_adapter=self._complexity_evaluation_dataset, model=model)
            else:
                self._estimate_in_subprocess(epoch, model, runner_config)

        self._reconcile_estimation(epoch)

    def _estimate_in_subprocess(
        self, epoch: int, model: PreTrainedModel, runner_config: ComplexityEstimationRunnerConfig
    ) -> None:
        # The flaky eval GPU dies with a native SIGSEGV mid-estimation; in-process that kills the
        # whole training run "without a trace". Run estimate() in a fresh, supervised subprocess
        # instead (mirrors Evaluator's per-unit isolation): a crash only kills the worker, which
        # supervise_unit restarts, and the runner resumes from its sibling .tmp parquet.
        out_path = self.out_path_for_epoch(epoch)

        # Persist the current weights once (reused across all restart attempts): the adapter via
        # save_pretrained, plus the trained thinking-token rows that PEFT's save_pretrained omits
        # (they live in the base model). The worker reloads both with the evaluator's LoRA path.
        model_dir = out_path.parent / "_model_snapshot"
        shutil.rmtree(model_dir, ignore_errors=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        # Unwrap a torch.compile wrapper so save_pretrained sees the real PeftModel.
        to_save = getattr(model, "_orig_mod", model)
        to_save.save_pretrained(model_dir.as_posix())
        assert (model_dir / "adapter_config.json").exists(), (
            f"save_pretrained wrote no adapter_config.json to {model_dir}"
        )
        if self._new_ids:
            write_thinking_token_rows(model, self._new_ids, model_dir)

        attn_implementation = getattr(model.config, "_attn_implementation", None)

        # Offload the live training model to CPU for the estimation window so the worker's own
        # base-model copy doesn't double VRAM against the resident parent. Restored in finally so a
        # crash/interrupt never leaves training on CPU. Disable with COMPLEXITY_OFFLOAD_PARENT=0.
        offload = os.environ.get("COMPLEXITY_OFFLOAD_PARENT") != "0"
        device = next(model.parameters()).device

        fd, spec_path = tempfile.mkstemp(prefix=f"complexity_spec_e{epoch}_", suffix=".pkl")
        try:
            with os.fdopen(fd, "wb") as f:
                try:
                    pickle.dump(
                        (
                            model_dir.as_posix(),
                            runner_config,
                            self._complexity_estimator,
                            self._complexity_evaluation_dataset,
                            attn_implementation,
                        ),
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                except Exception as ex:
                    raise RuntimeError(f"Failed to pickle complexity-estimation spec for epoch {epoch}: {ex}") from ex

            if offload:
                model.to("cpu")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            cmd = [sys.executable, _COMPLEXITY_WORKER, spec_path]
            child_env = {**os.environ, "COMPLEXITY_SUPERVISE": "0"}
            supervise_unit(
                cmd,
                child_env,
                label=f"epoch={epoch}",
                min_healthy_s=float(os.environ.get("COMPLEXITY_MIN_HEALTHY_S", "120")),
                max_fast=int(os.environ.get("COMPLEXITY_MAX_FAST_FAILURES", "3")),
                max_attempts=int(os.environ.get("COMPLEXITY_MAX_UNIT_ATTEMPTS", "50")),
                mem_watchdog_frac=float(os.environ.get("COMPLEXITY_MEM_WATCHDOG_FRAC", "0.92")),
                mem_poll_interval_s=float(os.environ.get("COMPLEXITY_MEM_WATCHDOG_INTERVAL_S", "2")),
                max_mem_kills=int(os.environ.get("COMPLEXITY_MEM_MAX_KILLS", "3")),
            )
        finally:
            if offload:
                model.to(device)
            try:
                os.unlink(spec_path)
            except OSError:
                pass

        if not out_path.exists():
            raise RuntimeError(f"complexity worker for epoch {epoch} exited 0 but wrote no parquet at {out_path}")
        # Estimation succeeded; the weight snapshot is no longer needed.
        shutil.rmtree(model_dir, ignore_errors=True)

    def _reconcile_estimation(self, epoch: int) -> None:
        # After a LoRA epoch the model may emit a thinking token instead of a single answer letter,
        # so those rows fail to measure (entropy_value is NaN). Reuse the previous epoch's
        # last-known-good entropy for the failed rows. Any residual NaN (epoch 0, or rows that
        # failed every epoch) is harmlessly dropped by the sampler (NaN scores sort to the bottom).
        path = self.out_path_for_epoch(epoch)
        df = pd.read_parquet(path)

        # The epoch to backfill from is the previous epoch that actually produced a dump, which is
        # `epoch - 1` only on the default every-epoch path. Under a sparse resampling_schedule the
        # intervening epochs have no parquet at all.
        prev_epoch = self._previous_resampling_epoch(epoch)

        if "entropy_value" not in df.columns:
            logger.warning(f"Complexity estimation parquet {path} has no entropy_value column; skipping reconcile.")
            return

        failed = df["entropy_value"].isna()
        failed_count = int(failed.sum())
        total_rows = len(df)
        failure_rate = float(failed.mean())

        # Too many failures is usually transient: the model drifts to chain-of-thought output for one
        # epoch, then recovers. Tolerate ONE such epoch -- the backfill below carries its failed rows on
        # the previous epoch's entropy. Two in a row means it is not recovering, so abort.
        exceeded = failure_rate > self._max_failed_estimation_fraction
        if exceeded:
            preamble = (
                f"Complexity estimation failed for {failure_rate:.1%} of rows at epoch {epoch} "
                f"(> {self._max_failed_estimation_fraction:.0%} allowed)"
            )
            if prev_epoch is None:
                raise RuntimeError(f"{preamble}; no previous epoch to backfill from, aborting training.")
            if self._previous_epoch_exceeded(prev_epoch):
                raise RuntimeError(
                    f"{preamble}, and epoch {prev_epoch} exceeded it too; the grace epoch did not "
                    f"recover. Aborting training."
                )
            logger.warning(
                f"{preamble}. Granting one grace epoch: backfilling from epoch {prev_epoch} and "
                f"continuing. Training aborts if the next resampling epoch also exceeds the threshold."
            )

        backfilled = 0
        has_backfill = prev_epoch is not None and bool(failed.any())
        if has_backfill:
            prev = pd.read_parquet(self.out_path_for_epoch(prev_epoch))
            prev_entropy = prev.dropna(subset=["entropy_value"]).set_index("question_id")["entropy_value"]
            df.loc[failed, "entropy_value"] = df.loc[failed, "question_id"].map(prev_entropy)
            backfilled = int(failed_count - df["entropy_value"].isna().sum())

        # Count how many rows the train sampler(s) will actually select from this finalized parquet
        # (after the non-positive-score drop). Computed here, after backfill, so `df` matches exactly
        # what SetResamplingPathCallback feeds the samplers for this same epoch.
        if self._train_dataset is not None:
            self._train_dataset.set_epoch(epoch)
            by_dataset = self._train_dataset.selected_sample_counts(df)
        else:
            by_dataset = {}
        selected_total = sum(by_dataset.values()) if by_dataset else None

        # A grace epoch must not hand training an empty sample: without this the collapse resurfaces
        # later as an opaque "all candidate rows scored <= 0" from BaseDatasetSampler.create_sample.
        # `selected_total` is None (not 0) when no train_dataset was wired up, so compare explicitly.
        # Raised before the parquet is rewritten below, so a resume re-reads the same unbackfilled
        # rows and reaches this same verdict instead of sailing past a now-low failure rate.
        if exceeded and selected_total == 0:
            raise RuntimeError(
                f"Grace epoch {epoch}: backfill left no rows with a positive score, so no training "
                f"samples would remain. Aborting training."
            )

        if has_backfill:
            df.to_parquet(path, index=False)
            logger.info(f"Backfilled {backfilled} failed entropy rows from epoch {prev_epoch}.")

        # Persist how many rows could not be measured this epoch (failed_count, before backfill).
        # This file is also the per-epoch "finalized" marker used by on_epoch_begin to skip
        # re-estimation on resume, so it is written last, only after a successful reconcile.
        self._write_estimation_stats(
            epoch,
            total_rows=total_rows,
            failed_rows=failed_count,
            backfilled=backfilled,
            residual_failed=int(df["entropy_value"].isna().sum()),
            selected_samples=selected_total,
            selected_samples_by_dataset=by_dataset or None,
            exceeded_threshold=exceeded,
        )

    def out_path_for_epoch(self, epoch: int) -> Path:
        return (
            self._out_path
            / str(epoch)
            / "complexity_estimation"
            / f"{self._complexity_evaluation_dataset.dataset.dataset_id}.parquet"
        )

    def _stats_path_for_epoch(self, epoch: int) -> Path:
        return self.out_path_for_epoch(epoch).parent / "estimation_stats.json"

    def resampling_epoch_for(self, epoch: int) -> int:
        """The epoch whose complexity dump (and therefore whose sample) `epoch` trains on.

        Without a schedule that is `epoch` itself: every epoch re-estimates and re-samples. With one
        it is the latest scheduled epoch <= `epoch`, so the sample stays frozen in between. Phrasing
        it as "latest <=" rather than "did this epoch resample" is what makes resume work: a run
        restarted at epoch 7 under schedule [0] sees on_epoch_begin(7) as its FIRST callback and
        must still resolve to epoch 0's parquet, which was written by the original run."""
        if self._resampling_schedule is None:
            return epoch
        eligible = [e for e in self._resampling_schedule if e <= epoch]
        assert eligible, f"No resampling epoch <= {epoch} in {self._resampling_schedule}; the schedule must start at 0."
        return max(eligible)

    def _previous_resampling_epoch(self, epoch: int) -> int | None:
        """The last epoch strictly before `epoch` that produced a dump, or None if there is none.

        `epoch - 1` on the default (every-epoch) path; under a sparse schedule the intervening
        epochs wrote no parquet, so backfill and the grace-epoch check must skip back to the real
        previous estimation instead."""
        if self._resampling_schedule is None:
            return epoch - 1 if epoch > 0 else None
        earlier = [e for e in self._resampling_schedule if e < epoch]
        return max(earlier) if earlier else None

    def _is_checkpoint_epoch(self, epoch: int) -> bool:
        # `epoch` == int(state.epoch) at on_epoch_begin == number of completed epochs. The previous
        # training epoch saved its checkpoint under trainer_state.epoch == epoch (when that value is in
        # save_schedule), so this tests whether the weights this complexity was measured on are
        # checkpointed and thus reloaded exactly on resume. No schedule => HF checkpoints every epoch
        # (save_strategy="epoch"), so every epoch qualifies.
        return self._save_schedule is None or epoch in self._save_schedule

    def _should_reuse(self, epoch: int) -> bool:
        # Reuse a finalized dump only when the weights it was measured on were checkpointed (so resume
        # reloads them exactly). Non-checkpointed epochs are re-trained -> the dump is stale -> recompute.
        return self._is_checkpoint_epoch(epoch) and self._stats_path_for_epoch(epoch).exists()

    def _previous_epoch_exceeded(self, prev_epoch: int) -> bool:
        # Read the grace bit off the previous epoch's finalized stats marker rather than tracking it in
        # memory: _should_reuse can skip reconcile entirely for a checkpointed epoch on resume, so an
        # instance counter would silently re-arm the grace after every restart. Takes the previous
        # *resampling* epoch (see _previous_resampling_epoch), not the current one.
        path = self._stats_path_for_epoch(prev_epoch)
        if not path.exists():
            # Pre-feature dump, or a data dir wiped between runs. Be permissive rather than fatal.
            return False
        return bool(json.loads(path.read_text()).get("exceeded_threshold", False))

    def _estimation_complete(self, epoch: int) -> bool:
        # estimate() writes out_path only at the very end and deletes its .tmp right after, so a
        # parquet with no sibling .tmp means estimation finished (reconcile may still be pending).
        out_path = self.out_path_for_epoch(epoch)
        return out_path.exists() and not out_path.with_suffix(".tmp").exists()

    def _write_estimation_stats(
        self,
        epoch: int,
        total_rows: int,
        failed_rows: int,
        backfilled: int,
        residual_failed: int,
        selected_samples: int | None = None,
        selected_samples_by_dataset: dict[str, int] | None = None,
        exceeded_threshold: bool = False,
    ) -> None:
        stats = {
            "epoch": epoch,
            "total_rows": total_rows,
            "failed_rows": failed_rows,
            "failure_rate": (failed_rows / total_rows) if total_rows else 0.0,
            "backfilled": backfilled,
            "residual_failed": residual_failed,
            # Whether this epoch blew past max_failed_estimation_fraction and was let through as the
            # one allowed grace epoch. Read back by _previous_epoch_exceeded on the next epoch.
            "exceeded_threshold": exceeded_threshold,
            # Number of rows the train sampler(s) actually select this epoch after dropping
            # non-positive scores (shrinks over time as the student catches the teacher).
            "selected_samples": selected_samples,
            "selected_samples_by_dataset": selected_samples_by_dataset,
        }
        path = self._stats_path_for_epoch(epoch)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(stats, indent=2))
        os.replace(tmp, path)


class SetResamplingPathCallback(TrainerCallback):
    def __init__(
        self,
        estimation_complexity_callback: EstimateComplexityCallback,
        resampling_ds: ResamplingDataset,
    ) -> None:
        super().__init__()

        self.estimation_complexity_callback = estimation_complexity_callback
        self.resampling_ds = resampling_ds

    @override
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        assert state.epoch is not None

        # Route through resampling_epoch_for so both callbacks agree on the effective epoch by
        # construction: on a frozen epoch EstimateComplexityCallback wrote nothing, and the dataset
        # must keep reading the last scheduled epoch's parquet.
        callback = self.estimation_complexity_callback
        self.resampling_ds.epoch = int(state.epoch)
        effective_epoch = callback.resampling_epoch_for(int(state.epoch))
        self.resampling_ds.selection_epoch = effective_epoch
        self.resampling_ds.dataset_path = callback.out_path_for_epoch(effective_epoch).as_posix()


class ResamplingTrainerConfig(LoRATrainerConfig):
    complexity_evaluation_dataset: QADatasetAdapter
    complexity_estimator: BaseComplexityEstimator
    complexity_estimation_runner_generation_config: ModelGenerateConfig
    # Abort training if more than this fraction of rows fail single-token entropy measurement
    # in an epoch (the model drifting to chain-of-thought output).
    max_failed_estimation_fraction: float = 0.1
    # Epochs at which the training sample is (re)drawn. None (the default) re-estimates complexity
    # and re-samples at the start of every epoch. A list freezes the sample between scheduled epochs
    # and skips estimation entirely on the frozen ones -- [0] draws once, on the untrained student,
    # and trains on that set for the whole run.
    resampling_schedule: list[int] | None = None
    # ResamplingDataset is iterable, so HF Trainer cannot attach its usual RandomSampler. Opt in
    # here to shuffle each finalized adapter sample independently and deterministically with
    # data_seed + training epoch, before a merged adapter concatenates its children.
    shuffle: bool = False

    @model_validator(mode="after")
    def _validate_resampling_schedule(self):
        if self.resampling_schedule is not None:
            assert self.resampling_schedule, "resampling_schedule must be non-empty (pass None to resample every epoch)"
            # Epoch 0 is the only epoch with no earlier dump to fall back on, so a schedule that
            # skips it would leave the dataset with no parquet to read at all.
            assert self.resampling_schedule[0] == 0, (
                f"resampling_schedule must start at epoch 0, got {self.resampling_schedule}"
            )
            assert self.resampling_schedule == sorted(set(self.resampling_schedule)), (
                f"resampling_schedule must be strictly increasing with no duplicates, got {self.resampling_schedule}"
            )
        return self

    @model_validator(mode="after")
    def _validate_packing_batching(self):
        # PackedResamplingDataset yields one variable-length block per row (real packs at
        # `budget`, null packs tiny); both its slot accounting and PackedSequenceCollator's
        # stack assume one block per device batch.
        if self.packing is not None:
            assert self.training_args.per_device_train_batch_size == 1, (
                "ResamplingTrainer with packing requires per_device_train_batch_size=1"
            )
        return self


class ResamplingTrainer(LoRATrainer[ResamplingTrainerConfig]):
    @override
    def train(self):
        try:
            return super().train()
        finally:
            # Aggregate per-epoch complexity-estimation failure counts. In finally so it also runs
            # after a partial (crashed/killed) run; the body is defensive and never re-raises.
            self._aggregate_estimation_stats()

    @override
    def _prepare_data(self):
        if self.config.packing is not None:
            # per_device_train_batch_size=1 (validated), so accumulation steps == the
            # docs-denominated effective batch resolved below.
            return PackedResamplingDataset(
                self.config.train_dataset,
                self.tokenizer,
                packing=self.config.packing,
                gradient_accumulation_steps=self.config.training_args.effective_train_batch_size
                // self.config.training_args.per_device_train_batch_size,
                shuffle=self.config.shuffle,
                data_seed=self.config.training_args.data_seed,
            )
        train_ds = ResamplingDataset(
            self.config.train_dataset,
            self.tokenizer,
            shuffle=self.config.shuffle,
            data_seed=self.config.training_args.data_seed,
        )
        return train_ds

    @override
    def _resolve_effective_train_batch_size(self) -> int:
        # PackedResamplingDataset keeps len() at the DOC count and null-pads each epoch up
        # to it, so the base packs/docs rescale must not apply: updates/epoch and docs per
        # update stay exactly as configured. This also bypasses the base _packing_stats
        # assert -- packing happens per epoch inside __iter__, not in _prepare_data.
        if self.config.packing is not None:
            return self.config.training_args.effective_train_batch_size
        return super()._resolve_effective_train_batch_size()

    @override
    def _build_trainer(self, train_ds):
        trainer = super()._build_trainer(train_ds)

        if self.config.packing is not None:
            # Null packs are harmless only on the token-normalized loss path: the per-chunk
            # num_items_in_batch divides a sum-CE, so an all-ignored block contributes exactly
            # zero. On the legacy mean-CE path each null block would be 0/0 = NaN and poison
            # the gradients -- fail at build time instead of at step one.
            assert trainer.model_accepts_loss_kwargs, (
                "Packed resampling requires the token-normalized loss path "
                "(Trainer.model_accepts_loss_kwargs); this base model's forward does not accept it."
            )

        self._estimate_complexity_callback = EstimateComplexityCallback(
            complexity_evaluation_dataset=self.config.complexity_evaluation_dataset,
            complexity_estimator=self.config.complexity_estimator,
            complexity_estimation_runner_generation_config=self.config.complexity_estimation_runner_generation_config,
            out_path=self._data_path,
            max_failed_estimation_fraction=self.config.max_failed_estimation_fraction,
            save_schedule=self.config.save_schedule,
            train_dataset=self.config.train_dataset,
            # Set by LoRATrainer.model when train_thinking_token_embeddings is on (available here
            # because super()._build_trainer already constructed the model). Lets the estimation
            # subprocess persist/reload the trained <think> rows that PEFT's save_pretrained omits.
            new_ids=self._snapshot.new_ids if self._snapshot else None,
            resampling_schedule=self.config.resampling_schedule,
        )
        trainer.add_callback(self._estimate_complexity_callback)
        trainer.add_callback(
            SetResamplingPathCallback(
                estimation_complexity_callback=self._estimate_complexity_callback, resampling_ds=train_ds
            )
        )

        return trainer

    def _path_for_epoch(self, epoch: int) -> Path:
        return self._data_path / str(epoch)

    @property
    def _data_path(self) -> Path:
        return Path(self.config.out_path) / "resampling_trainer_data"

    def _aggregate_estimation_stats(self) -> None:
        # Roll the per-epoch estimation_stats.json files up into one summary keyed by epoch. Runs in
        # train()'s finally, so it must never raise (it would mask a training error) — log and move on.
        try:
            base = self._data_path
            if not base.exists():
                return

            dataset_id = self.config.complexity_evaluation_dataset.dataset.dataset_id

            details: list[dict] = []
            for epoch_dir in sorted(
                (d for d in base.iterdir() if d.is_dir() and d.name.isdigit()),
                key=lambda d: int(d.name),
            ):
                epoch = int(epoch_dir.name)
                stats_path = epoch_dir / "complexity_estimation" / "estimation_stats.json"
                if stats_path.exists():
                    entry = json.loads(stats_path.read_text())
                    entry["source"] = "recorded"
                    # Pre-feature stats files lack selected_samples; surface as None rather than KeyError.
                    entry.setdefault("selected_samples", None)
                    entry.setdefault("selected_samples_by_dataset", None)
                    entry.setdefault("exceeded_threshold", None)
                else:
                    # A dump produced before this feature: its pre-backfill count is gone (reconcile
                    # overwrote the parquet), so report the post-backfill residual as a best effort.
                    parquet = epoch_dir / "complexity_estimation" / f"{dataset_id}.parquet"
                    if not parquet.exists():
                        continue
                    col = pd.read_parquet(parquet, columns=["entropy_value"])["entropy_value"]
                    residual = int(col.isna().sum())
                    entry = {
                        "epoch": epoch,
                        "total_rows": len(col),
                        "failed_rows": residual,
                        "failure_rate": (residual / len(col)) if len(col) else 0.0,
                        "backfilled": None,
                        "residual_failed": residual,
                        # Can't recompute reliably here: this aggregate only knows the complexity
                        # dataset_id, not the train adapters' top_k(s)/scoring. Leave unknown.
                        "selected_samples": None,
                        "selected_samples_by_dataset": None,
                        # The grace bit lived only in the stats file; the parquet cannot recover it.
                        "exceeded_threshold": None,
                        "source": "derived",
                    }
                details.append(entry)

            summary = {
                "total_failed_rows": sum(d["failed_rows"] for d in details),
                "failed_rows_by_epoch": {str(d["epoch"]): d["failed_rows"] for d in details},
                "selected_samples_by_epoch": {str(d["epoch"]): d.get("selected_samples") for d in details},
                # Epochs that were let through as a grace epoch -- their scores are partly stale.
                "grace_epochs": [d["epoch"] for d in details if d.get("exceeded_threshold")],
                "details": details,
            }
            out_path = base / "complexity_estimation_failures.json"
            tmp = out_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(summary, indent=2))
            os.replace(tmp, out_path)
            logger.info(f"Wrote complexity-estimation failure summary to {out_path}")
        except Exception as ex:
            logger.warning(f"Failed to aggregate complexity-estimation stats: {ex}")
