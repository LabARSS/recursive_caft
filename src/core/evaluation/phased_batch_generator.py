import gc
import json
import os
import shutil
import tempfile
import time
import types
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import psutil
import torch
import torch.nn.functional as F
import transformers.modeling_flash_attention_utils as _flash_utils
from tqdm import tqdm
from transformers import DynamicCache, PreTrainedModel, PreTrainedTokenizer
from transformers.cache_utils import _static_cache_update

from core.utils.logger import logger


def _malloc_trim() -> None:
    """Ask glibc to release free arena pages back to the OS.

    No-op on non-glibc platforms (macOS, Alpine/musl). Linux-only.
    """
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _pin_glibc_mmap_threshold() -> None:
    """Pin glibc's M_MMAP_THRESHOLD low and disable adaptive growth.

    Default glibc grows M_MMAP_THRESHOLD up to 32MB after the first large
    mmap'd chunk is freed, after which same-sized allocations land on the
    heap (where free only enlarges glibc's freelists). Our per-layer KV
    staging produces lots of ~13MB tensors that fall into this trap and
    cause a multi-tens-of-GB gap between tracked KV and process RSS.

    Pinning the threshold via mallopt forces every alloc >128KB through
    mmap/munmap, so freeing returns pages to the OS immediately.

    Note: do NOT touch M_MMAP_MAX. The glibc default (65536) is plenty;
    setting it to 0 actually *disables* mmap entirely, which is the
    opposite of what we want.

    No-op on non-glibc platforms.
    """
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        # malloc.h: M_MMAP_THRESHOLD = -3
        libc.mallopt(-3, 128 * 1024)
    except (OSError, AttributeError):
        pass


_pin_glibc_mmap_threshold()


def _mem_snapshot(staged_slots: int | None = None) -> str:
    rss_gb = psutil.Process().memory_info().rss / 1e9
    parts = [f"rss={rss_gb:.2f}GB"]
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        alloc = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        parts.extend(
            [
                f"cuda_free={free / 1e9:.2f}GB",
                f"cuda_alloc={alloc / 1e9:.2f}GB",
                f"cuda_reserved={reserved / 1e9:.2f}GB",
                f"cuda_total={total / 1e9:.2f}GB",
            ]
        )
    if staged_slots is not None:
        parts.append(f"staged_slots={staged_slots}")
    return " ".join(parts)


@dataclass
class GenerationResult:
    sequences: list[list[int]]
    num_truncated: int
    total: int
    truncated: list[bool] = field(default_factory=list)
    thinking_budget_exhausted: list[bool] = field(default_factory=list)


@dataclass
class _Slot:
    index: int
    prompt_ids: list[int]
    prompt_len: int
    generated_ids: list[int] = field(default_factory=list)
    seq_position: int = 0  # total tokens seen = prompt_len + len(generated_ids)
    in_thinking: bool = False
    thinking_budget_exhausted: bool = False


@dataclass
class _StagedSlot:
    """Slot with KV cache staged off the GPU.

    Either RAM-resident (key_cache/value_cache set, spill_path None) or
    disk-resident (spill_path set, key_cache/value_cache None). Created and
    consumed exclusively through _StagedKVStore.
    """

    slot: _Slot
    valid_len: int
    nbytes: int  # total CPU bytes of this slot's K+V tensors
    # Per-layer CPU tensors: shape [1, num_kv_heads, valid_len, head_dim].
    # None when the slot has been spilled to disk.
    key_cache: list[torch.Tensor] | None = None
    value_cache: list[torch.Tensor] | None = None
    # Path of the on-disk file when spilled; None when RAM-resident.
    spill_path: str | None = None

    @property
    def on_disk(self) -> bool:
        return self.spill_path is not None


class _StagedKVStore:
    """Owns CPU-staged KV slots for one generate() call, with disk spill.

    Tracks the cumulative RAM footprint of staged KV. When staging a slot would
    push the footprint over `threshold_bytes`, that slot's KV is written to one
    file on disk instead of RAM (watermark spill: RAM-resident slots are never
    evicted). Restore reads the slot back — from RAM or disk — and deletes the
    file for disk slots.

    Disk format: the per-layer K and V tensors are each stacked into one
    contiguous tensor and saved together via torch.save — one file per slot,
    avoiding per-layer pickle overhead and many small writes.
    """

    def __init__(self, threshold_bytes: int, spill_parent_dir: str | None) -> None:
        self.threshold_bytes = threshold_bytes
        if spill_parent_dir is not None:
            os.makedirs(spill_parent_dir, exist_ok=True)
        # Fresh unique subdir; the store fully owns and removes it on close().
        self.spill_dir = tempfile.mkdtemp(prefix="kv_spill_", dir=spill_parent_dir)
        self._ram_bytes = 0
        self._spill_seq = 0
        self._spilled_count = 0
        self._spilled_bytes = 0
        logger.info(f"[kv-store] init threshold={threshold_bytes / 1e9:.1f}GB spill_dir={self.spill_dir}")

    @staticmethod
    def _kv_nbytes(keys: list[torch.Tensor], vals: list[torch.Tensor]) -> int:
        """Total CPU bytes of a slot's per-layer K and V tensors."""
        return sum(t.numel() * t.element_size() for t in keys) + sum(t.numel() * t.element_size() for t in vals)

    def stage(
        self,
        slot: _Slot,
        valid_len: int,
        keys: list[torch.Tensor],
        vals: list[torch.Tensor],
    ) -> _StagedSlot:
        """Build a _StagedSlot, spilling to disk when tracked KV bytes would exceed the cap."""
        nbytes = self._kv_nbytes(keys, vals)
        if self._ram_bytes + nbytes > self.threshold_bytes:
            path = self._spill_to_disk(keys, vals)
            self._spilled_count += 1
            self._spilled_bytes += nbytes
            rss_gb = psutil.Process().memory_info().rss / 1e9
            logger.trace(
                f"[kv-store] spill slot={slot.index} valid_len={valid_len} "
                f"nbytes={nbytes / 1e9:.3f}GB ram_kv={self._ram_bytes / 1e9:.2f}GB "
                f"rss={rss_gb:.2f}GB"
            )
            return _StagedSlot(slot=slot, valid_len=valid_len, nbytes=nbytes, spill_path=path)
        self._ram_bytes += nbytes
        return _StagedSlot(slot=slot, valid_len=valid_len, nbytes=nbytes, key_cache=keys, value_cache=vals)

    def restore(self, staged: _StagedSlot) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return (keys, vals) CPU tensors; read+delete the file for disk slots."""
        if staged.spill_path is not None:
            keys, vals = self._read_from_disk(staged.spill_path)
            os.remove(staged.spill_path)
            return keys, vals
        self._ram_bytes -= staged.nbytes
        keys, vals = staged.key_cache, staged.value_cache
        staged.key_cache = staged.value_cache = None
        return keys, vals

    def _spill_to_disk(self, keys: list[torch.Tensor], vals: list[torch.Tensor]) -> str:
        """Stack per-layer tensors and write one file. Returns the path."""
        # torch.stack on a new dim 0 -> contiguous [num_layers, 1, kv_heads, valid_len, head_dim].
        k_stacked = torch.stack(keys, dim=0)
        v_stacked = torch.stack(vals, dim=0)
        path = os.path.join(self.spill_dir, f"kv_{self._spill_seq:08d}.pt")
        self._spill_seq += 1
        try:
            torch.save({"k": k_stacked, "v": v_stacked}, path)
        except Exception:
            # torch.save's zip-write errors (`unexpected pos X vs Y`,
            # `basic_ios::clear: iostream error`) almost always mean disk-full
            # or I/O fault. Surface enough state to tell which without needing
            # the formatter's locals renderer (which itself fails on tensors).
            free_bytes = shutil.disk_usage(self.spill_dir).free
            nbytes = (
                k_stacked.numel() * k_stacked.element_size()
                + v_stacked.numel() * v_stacked.element_size()
            )
            logger.error(
                f"[kv-store] spill write failed path={path} "
                f"size={nbytes / 1e9:.2f}GB "
                f"spill_dir_free={free_bytes / 1e9:.2f}GB "
                f"spilled_so_far={self._spilled_bytes / 1e9:.2f}GB "
                f"spilled_count={self._spilled_count}"
            )
            raise
        return path

    @staticmethod
    def _read_from_disk(path: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Load a spilled file and unstack back into per-layer lists."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        k, v = payload["k"], payload["v"]
        return [k[i] for i in range(k.shape[0])], [v[i] for i in range(v.shape[0])]

    def close(self) -> None:
        """Delete the spill directory. Safe to call multiple times."""
        logger.info(
            f"[kv-store] close spilled_slots={self._spilled_count} spilled_bytes={self._spilled_bytes / 1e9:.2f}GB"
        )
        shutil.rmtree(self.spill_dir, ignore_errors=True)
        _malloc_trim()


class _PrefillCacheView(DynamicCache):
    """Lightweight cache view for prefill forward passes.

    Uses _static_cache_update (shared cache_position) with no branches,
    so torch.compile can trace a single code path per layer.
    """

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ):
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs else None
        _static_cache_update(
            self.key_cache[layer_idx],
            self.value_cache[layer_idx],
            key_states,
            value_states,
            cache_position,
        )
        seq_end = self._active_seq_len + 1
        return (
            self.key_cache[layer_idx][:, :, :seq_end, :],
            self.value_cache[layer_idx][:, :, :seq_end, :],
        )

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self._active_seq_len


class _PreAllocatedBatchCache(DynamicCache):
    """Pre-allocated KV cache for batched decode with per-row positions.

    Uses a branchless update() (always per-row indexed write) so
    torch.compile sees a single code path and avoids guard explosion.
    Prefill uses a separate _PrefillCacheView returned by prefill_view().
    """

    is_compileable = True

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_batch_size: int,
        max_cache_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.max_cache_len = max_cache_len
        self._active_seq_len = 0
        self._per_row_cache_positions = torch.zeros(max_batch_size, dtype=torch.long, device=device)
        self._batch_indices = torch.arange(max_batch_size, device=device)
        cache_shape = (max_batch_size, num_kv_heads, max_cache_len, head_dim)
        for _ in range(num_layers):
            k = torch.zeros(cache_shape, dtype=dtype, device=device)
            v = torch.zeros(cache_shape, dtype=dtype, device=device)
            torch._dynamo.mark_static_address(k)
            torch._dynamo.mark_static_address(v)
            self.key_cache.append(k)
            self.value_cache.append(v)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ):
        k_cache = self.key_cache[layer_idx]
        v_cache = self.value_cache[layer_idx]
        k_cache[self._batch_indices, :, self._per_row_cache_positions, :] = key_states[:, :, 0, :]
        v_cache[self._batch_indices, :, self._per_row_cache_positions, :] = value_states[:, :, 0, :]
        seq_end = self._active_seq_len + 1
        return (
            self.key_cache[layer_idx][:, :, :seq_end, :],
            self.value_cache[layer_idx][:, :, :seq_end, :],
        )

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self._active_seq_len

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len

    def prefill_view(self, batch_idx: int, seq_len: int) -> _PrefillCacheView:
        """Return a single-slot cache wrapper that writes directly into row `batch_idx`."""
        view = _PrefillCacheView()
        view._active_seq_len = seq_len - 1  # update() will see seq_end = seq_len
        view.key_cache = [self.key_cache[i][batch_idx : batch_idx + 1] for i in range(len(self))]
        view.value_cache = [self.value_cache[i][batch_idx : batch_idx + 1] for i in range(len(self))]
        return view

    def stage_row_to_cpu(self, batch_idx: int, valid_len: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Copy one batch row's KV data to CPU tensors."""
        keys = [self.key_cache[i][batch_idx : batch_idx + 1, :, :valid_len, :].cpu() for i in range(len(self))]
        vals = [self.value_cache[i][batch_idx : batch_idx + 1, :, :valid_len, :].cpu() for i in range(len(self))]
        return keys, vals

    def restore_row_from_cpu(
        self, batch_idx: int, valid_len: int, keys: list[torch.Tensor], vals: list[torch.Tensor]
    ) -> None:
        """Copy CPU-staged KV data back into a batch row on GPU."""
        for i in range(len(self)):
            self.key_cache[i][batch_idx, :, :valid_len, :] = keys[i][0].to(self.key_cache[i].device)
            self.value_cache[i][batch_idx, :, :valid_len, :] = vals[i][0].to(self.value_cache[i].device)


class BatchGenerator:
    """Token-by-token generation with static batching via model.forward().

    All prompts are prefilled upfront (one-by-one) with KV cache staged on
    CPU, producing a uniform queue of _StagedSlots. Decode phases then
    restore slots to GPU, decode in batches, and re-stage promoted slots.

    Generation is split into phases of _PHASE_STEP tokens each. Unfinished
    sequences are promoted to the next phase by staging their KV cache on
    CPU — no re-prefill needed.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        max_new_tokens: int,
        max_batch_size: int = 8,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        max_thinking_tokens: int | None = None,
        thinking_end_token_id: int | None = None,
        kv_cache_offload_threshold_gb: float = 120.0,
        kv_cache_spill_dir: str | None = None,
    ):
        self.model = model
        # Use uncompiled model for prefill, compiled for decode — mirrors HF generate().
        if hasattr(model, "_orig_mod"):
            self._compiled_model = model
            self._uncompiled_model = model._orig_mod
        else:
            self._compiled_model = model
            self._uncompiled_model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.max_batch_size = max_batch_size
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_thinking_tokens = max_thinking_tokens
        self.thinking_end_token_id = thinking_end_token_id
        self._kv_offload_threshold_bytes = int(kv_cache_offload_threshold_gb * 1e9)
        self._kv_spill_dir = kv_cache_spill_dir
        self._enforce_thinking_cap = max_thinking_tokens is not None and thinking_end_token_id is not None
        if (max_thinking_tokens is None) != (thinking_end_token_id is None):
            logger.warning(
                "BatchGenerator: max_thinking_tokens and thinking_end_token_id must both be set "
                "to enforce a thinking cap; got max_thinking_tokens=%r, thinking_end_token_id=%r. "
                "Cap will not be enforced.",
                max_thinking_tokens,
                thinking_end_token_id,
            )

        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self._effective_batch_size = max_batch_size
        self._cache = None
        self._vram_safe_margin = 0.10  # reserve 10% of total VRAM as buffer
        self._vram_check_interval = 50  # periodic free-VRAM safety check every N decode steps
        self._vram_reduced_bs: int | None = None  # sticky reduced bs after VRAM pressure
        self._usable_vram: int | None = None  # measured once after prefill
        self._current_max_seqlen: int = 0  # set before each forward call for _get_unpad_data

        self._patch_causal_mask(self._uncompiled_model)
        self._install_unpad_cache()

    @staticmethod
    def _patch_causal_mask(model: PreTrainedModel) -> None:
        """Remove the right-padding check from _update_causal_mask for FA2 compatibility.

        Multiple HF models (Qwen2, Phi4, etc.) raise ValueError when they detect
        right-padded attention masks with FA2.  The check is a guard — FA2's unpadding
        works correctly with right-padded masks.  This patch keeps the FA2 mask logic
        (return mask when it has zeros, else None) but removes the ValueError.

        Walks through wrapper layers (PEFT/LoRA add an extra .model level) to find
        the inner transformer that owns _update_causal_mask.
        """
        # Walk through known wrapper layers to find the object with _update_causal_mask.
        # Plain:   model.model  (e.g. Qwen2ForCausalLM.model → Qwen2Model)
        # PEFT:    model.model.model  (PeftModel → LoraModel → XForCausalLM → XModel)
        candidate = model
        for _ in range(3):  # up to 3 levels of .model
            if not hasattr(candidate, "model"):
                break
            candidate = candidate.model
            if hasattr(candidate, "_update_causal_mask"):
                break
        else:
            return

        if not hasattr(candidate, "_update_causal_mask"):
            return

        original = candidate._update_causal_mask

        def _update_causal_mask(
            self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions=False
        ):
            if self.config._attn_implementation == "flash_attention_2":
                # Always return the mask for FA2. In batched decode the mask
                # always contains zeros (padding) so the torch.any() check was
                # redundant and caused a GPU→CPU sync + torch.compile graph break.
                return attention_mask
            return original(attention_mask, input_tensor, cache_position, past_key_values, output_attentions)

        candidate._update_causal_mask = types.MethodType(_update_causal_mask, candidate)

    def _install_unpad_cache(self) -> None:
        """Patch _get_unpad_data to cache per data_ptr and use CPU-known max_seqlen.

        HF calls _get_unpad_data once per decoder layer (28× per forward pass)
        with the *same* attention mask. The original does nonzero + .item() each
        time — 28 GPU→CPU syncs. This patch computes once and returns cached
        results for subsequent layers. max_seqlen is taken from
        self._current_max_seqlen (set in _batched_decode from Python-known
        valid_lens) to avoid the .item() GPU→CPU sync entirely.
        """
        original = _flash_utils._get_unpad_data
        generator = self
        _cache: dict = {}

        def _cached_get_unpad_data(attention_mask: torch.Tensor):
            key = attention_mask.data_ptr()
            if key in _cache:
                return _cache[key]
            seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            max_seqlen = generator._current_max_seqlen
            cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
            result = (indices, cu_seqlens, max_seqlen)
            _cache[key] = result
            return result

        def _clear():
            _cache.clear()

        _cached_get_unpad_data.clear_cache = _clear
        _flash_utils._get_unpad_data = _cached_get_unpad_data
        self._unpad_clear = _clear
        self._original_get_unpad_data = original

    def _init_cache(self, max_seq_len: int, batch_size: int) -> _PreAllocatedBatchCache:
        config = self.model.config
        num_layers = config.num_hidden_layers
        num_kv_heads = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        device = self.model.device
        dtype = self.model.dtype
        return _PreAllocatedBatchCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_batch_size=batch_size,
            max_cache_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

    def _check_vram_pressure(self) -> bool:
        """Return True if free VRAM is below the safety margin."""
        if not torch.cuda.is_available():
            return False
        free, total = torch.cuda.mem_get_info()
        return free < total * self._vram_safe_margin

    def _estimate_cache_bytes(self, max_cache_len: int, batch_size: int) -> int:
        """Estimate GPU memory (bytes) needed for a KV cache allocation."""
        config = self.model.config
        num_layers = config.num_hidden_layers
        num_kv_heads = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        element_size = torch.tensor([], dtype=self.model.dtype).element_size()
        # Per layer: 2 tensors (K + V), each of shape (batch_size, num_kv_heads, max_cache_len, head_dim)
        per_layer = 2 * batch_size * num_kv_heads * max_cache_len * head_dim * element_size
        return num_layers * per_layer

    @staticmethod
    def _snap_to_even(bs: int) -> int:
        """Round bs down to the nearest even number, but never below 1.

        Warp-aligned bs keeps decode kernels happy with no real throughput cost
        (worst case: one slot dropped).
        """
        if bs <= 1:
            return 1
        return bs - (bs % 2)

    def _fit_batch_size_to_vram(self, max_cache_len: int, batch_size: int) -> int:
        """Reduce batch_size until the estimated KV cache fits in usable VRAM.

        Uses self._usable_vram (measured once after prefill). Pure math, no
        CUDA queries. No-ops if _usable_vram was not measured (CPU-only).

        Steps down by ~25% per iteration (floor 2) instead of halving — keeps
        successive phases on the same bs when seq_len growth is modest.
        """
        if self._usable_vram is None:
            return batch_size

        while batch_size >= 1:
            needed = self._estimate_cache_bytes(max_cache_len, batch_size)
            if needed <= self._usable_vram:
                return batch_size
            if batch_size == 1:
                break
            step = max(2, batch_size // 4)
            new_bs = self._snap_to_even(max(batch_size - step, 1))
            logger.info(
                f"[vram] Cache for bs={batch_size}, seq_len={max_cache_len} needs "
                f"{needed / 1e9:.2f} GB but only {self._usable_vram / 1e9:.2f} GB usable. "
                f"Reducing batch size to {new_bs}."
            )
            batch_size = new_bs

        needed = self._estimate_cache_bytes(max_cache_len, 1)
        raise RuntimeError(
            f"[vram] KV cache for batch_size=1, seq_len={max_cache_len} requires "
            f"{needed / 1e9:.2f} GB but only {self._usable_vram / 1e9:.2f} GB is available. "
            f"Cannot proceed."
        )

    def _measure_usable_vram(self) -> None:
        """Measure free VRAM once and store as self._usable_vram."""
        if not torch.cuda.is_available():
            return
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        self._usable_vram = free - int(total * self._vram_safe_margin)

    _PHASE_STEP = 256  # max tokens generated per phase

    @torch.no_grad()
    def generate(self, prompts: list[list[int]], checkpoint_path: str | None = None) -> GenerationResult:
        """Generate responses for a list of prompts using phased generation.

        All prompts are prefilled upfront (one-by-one, batch_size=1) with KV
        cache staged on CPU. Then decode phases restore slots to GPU, decode
        in batches, and re-stage promoted slots to CPU for the next phase.

        Args:
            prompts: List of token ID sequences (one per sample).

        Returns:
            GenerationResult with generated sequences and truncation stats.
        """
        results: list[list[int] | None] = [None] * len(prompts)
        truncated_flags: list[bool] = [False] * len(prompts)
        thinking_exhausted_flags: list[bool] = [False] * len(prompts)
        pbar = tqdm(total=len(prompts), desc="Generating")
        max_prompt_len = max(len(p) for p in prompts)
        max_total = max_prompt_len + self.max_new_tokens

        # --- Resume from a per-chunk checkpoint, if one exists ---
        # Each completed phase persists finished results + in-flight token ids
        # (no KV — see _write_checkpoint). On resume we re-prefill the unfinished
        # rows from prompt + generated_ids to rebuild their KV and continue, so a
        # crash/relaunch loses minutes (one re-prefill), not the whole chunk.
        ckpt = self._load_checkpoint(checkpoint_path)
        resume_rows: dict = ckpt["rows"] if ckpt else {}
        resume_phase: int | None = ckpt["phase"] if ckpt else None

        # Per prompt: already-done rows seed `results`; the rest need (re-)prefill.
        # prefill_plan entries: (index, prefill_ids, generated_ids | None).
        prefill_plan: list[tuple[int, list[int], list[int] | None]] = []
        for i, prompt_ids in enumerate(prompts):
            row = resume_rows.get(str(i))
            if row is not None and row["done"]:
                results[i] = row["generated_ids"]
                truncated_flags[i] = row["truncated"]
                thinking_exhausted_flags[i] = row["thinking_budget_exhausted"]
                pbar.update(1)
            elif row is not None and row["generated_ids"]:
                # In-flight: rebuild KV for prompt + all-but-last generated token,
                # then resume decoding from the last generated token (see _prefill).
                gen = row["generated_ids"]
                prefill_plan.append((i, prompt_ids + gen[:-1], gen))
            else:
                prefill_plan.append((i, prompt_ids, None))

        if ckpt is not None:
            logger.info(
                f"[ckpt] resuming {checkpoint_path}: completed_phase={resume_phase}, "
                f"{sum(1 for r in results if r is not None)} done, {len(prefill_plan)} to continue"
            )

        # --- (Re-)prefill the unfinished prompts (batch_size=1 temp cache) ---
        max_prefill_len = max((len(pre) for _, pre, _ in prefill_plan), default=1)
        prefill_cache = self._init_cache(max_prefill_len + 1, 1)
        slot_queue: deque[_StagedSlot] = deque()

        # Owns the CPU-staged KV for this call and spills overflow to disk once
        # the staged footprint crosses the threshold. Torn down in finally.
        kv_store = _StagedKVStore(self._kv_offload_threshold_bytes, self._kv_spill_dir)
        try:
            prefill_start = time.perf_counter()
            total_prefill_tokens = 0
            for i, prefill_ids, gen in tqdm(prefill_plan, desc="Prefilling", leave=False):
                slot = _Slot(
                    index=i,
                    prompt_ids=prompts[i],
                    prompt_len=len(prompts[i]),
                    in_thinking=self._enforce_thinking_cap,
                )
                self._cache = prefill_cache
                self._valid_lens = [0]
                self._prefill(slot, prefill_ids, 0, resume_generated_ids=gen)
                total_prefill_tokens += len(prefill_ids)
                if gen is not None and self._enforce_thinking_cap:
                    # Reconstruct thinking-cap state from the generated tokens.
                    slot.in_thinking = self.thinking_end_token_id not in gen
                    slot.thinking_budget_exhausted = resume_rows[str(i)]["thinking_budget_exhausted"]

                # Stage KV off the GPU (RAM, or disk once over threshold)
                valid_len = self._valid_lens[0]
                keys, vals = prefill_cache.stage_row_to_cpu(0, valid_len)
                slot_queue.append(kv_store.stage(slot, valid_len, keys, vals))

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            prefill_time = time.perf_counter() - prefill_start
            logger.info(
                f"[perf] Prefill: {len(prefill_plan)} prompts, {total_prefill_tokens} tokens, "
                f"{prefill_time:.4f}s ({total_prefill_tokens / max(prefill_time, 1e-9):.0f} tok/s)"
            )

            # Free prefill cache
            self._cache = None
            prefill_cache = None
            self._measure_usable_vram()

            # --- Phase loop ---
            logger.info(
                f"[phase] Starting generation: {len(slot_queue)} prompts, "
                f"phase_step={self._PHASE_STEP} max={self.max_new_tokens}"
            )

            phase = (resume_phase + 1) if resume_phase is not None else 0
            total_threshold = min(self._PHASE_STEP * (phase + 1), max_total)

            while True:
                if not slot_queue:
                    break  # nothing left to generate (e.g. a fully-resumed chunk)
                promote_queue: deque[_StagedSlot] = deque()

                logger.info(
                    f"[phase] Starting phase {phase + 1}: {len(slot_queue)} sequences, "
                    f"total_threshold={total_threshold}"
                )
                logger.trace(
                    f"[trace] phase_start phase={phase + 1} staged={len(slot_queue)} "
                    f"total_threshold={total_threshold} {_mem_snapshot(staged_slots=len(slot_queue))}"
                )

                is_last = total_threshold >= max_total
                self._run_phase(
                    slot_queue,
                    kv_store,
                    results,
                    truncated_flags,
                    thinking_exhausted_flags,
                    total_threshold,
                    promote_queue,
                    pbar,
                    total_threshold + 1,
                    is_last,
                )

                self._cache = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _malloc_trim()
                logger.trace(
                    f"[trace] phase_end phase={phase + 1} promoted={len(promote_queue)} "
                    f"{_mem_snapshot(staged_slots=len(promote_queue))}"
                )

                # Persist progress for crash-resume (finished results + in-flight
                # token ids; no KV). Cheap and atomic — see _write_checkpoint.
                self._write_checkpoint(
                    checkpoint_path, phase, results, truncated_flags, thinking_exhausted_flags, promote_queue
                )

                if not promote_queue:
                    break

                slot_queue = promote_queue
                phase += 1
                total_threshold = min(self._PHASE_STEP * (phase + 1), max_total)
                logger.info(f"[phase] Phase {phase}: {len(slot_queue)} sequences promoted")

            pbar.close()
            sequences = [r if r is not None else [] for r in results]
            return GenerationResult(
                sequences=sequences,
                # Derive from flags (not the per-run `trunc` counter) so a resumed
                # chunk also counts truncations that happened before the crash.
                num_truncated=sum(truncated_flags),
                total=len(prompts),
                truncated=truncated_flags,
                thinking_budget_exhausted=thinking_exhausted_flags,
            )
        finally:
            kv_store.close()

    @staticmethod
    def _load_checkpoint(checkpoint_path: str | None) -> dict | None:
        """Load a per-chunk resume checkpoint, or None if absent/unreadable.

        A corrupt/partial checkpoint is treated as absent (the chunk runs fresh),
        mirroring Evaluator._load_chunk's tolerance for a torn cache file.
        """
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return None
        try:
            with open(checkpoint_path) as f:
                ckpt = json.load(f)
            if not isinstance(ckpt.get("phase"), int) or not isinstance(ckpt.get("rows"), dict):
                raise ValueError("missing/invalid 'phase' or 'rows'")
            required = ("generated_ids", "done", "truncated", "thinking_budget_exhausted")
            for row in ckpt["rows"].values():
                if not all(k in row for k in required):
                    raise ValueError("row missing required keys")
            return ckpt
        except (OSError, ValueError, TypeError) as ex:  # JSONDecodeError is a ValueError
            logger.warning(f"[ckpt] ignoring unreadable checkpoint {checkpoint_path}: {ex}; running chunk fresh")
            return None

    @staticmethod
    def _write_checkpoint(
        checkpoint_path: str | None,
        phase: int,
        results: list[list[int] | None],
        truncated_flags: list[bool],
        thinking_exhausted_flags: list[bool],
        promote_queue: "deque[_StagedSlot]",
    ) -> None:
        """Persist finished results + in-flight token ids after a phase (no KV).

        Token ids only (a few MB), written atomically (tmp + os.replace) once per
        phase — negligible next to the multi-second per-phase KV restores. Lets a
        re-launched process resume this chunk via generate(checkpoint_path=...).
        """
        if not checkpoint_path:
            return
        rows: dict[str, dict] = {}
        for i, gen in enumerate(results):
            if gen is not None:
                rows[str(i)] = {
                    "generated_ids": gen,
                    "done": True,
                    "truncated": bool(truncated_flags[i]),
                    "thinking_budget_exhausted": bool(thinking_exhausted_flags[i]),
                }
        for staged in promote_queue:
            s = staged.slot
            rows[str(s.index)] = {
                "generated_ids": s.generated_ids,
                "done": False,
                "truncated": False,
                "thinking_budget_exhausted": bool(s.thinking_budget_exhausted),
            }
        tmp = f"{checkpoint_path}.tmp"
        try:
            with open(tmp, "w") as f:
                json.dump({"phase": phase, "rows": rows}, f)
            os.replace(tmp, checkpoint_path)
        except OSError as ex:
            logger.warning(f"[ckpt] failed to write checkpoint {checkpoint_path}: {ex}")

    def _run_phase(
        self,
        slot_queue: deque[_StagedSlot],
        kv_store: _StagedKVStore,
        results: list[list[int] | None],
        truncated_flags: list[bool],
        thinking_exhausted_flags: list[bool],
        total_threshold: int,
        promote_queue: deque[_StagedSlot],
        pbar: tqdm,
        max_cache_len: int,
        is_last: bool,
    ) -> int:
        """Run a complete phase with static batching.

        Pops staged slots from slot_queue, restores KV (RAM or disk) → GPU,
        decodes until done or budget exhausted, re-stages promoted slots
        through kv_store.

        Returns the number of truncated sequences.
        """
        self._cache = None

        max_bs = self._vram_reduced_bs if self._vram_reduced_bs is not None else self.max_batch_size
        effective_bs = min(len(slot_queue), max_bs)

        # Pre-allocation VRAM check: reduce batch size until cache fits (pure math)
        effective_bs = self._fit_batch_size_to_vram(max_cache_len, effective_bs)

        # Track the slot count directly. mark_dynamic on bs made phase-to-phase
        # bs changes free, so the prior hysteresis policy (suppress changes <25%)
        # only served to waste compute on padding rows. Just snap to even for
        # warp-aligned kernels.
        effective_bs = self._snap_to_even(effective_bs)

        if effective_bs != self._effective_batch_size:
            logger.info(f"[perf] Adjusting batch size: {self._effective_batch_size} → {effective_bs}.")
            self._effective_batch_size = effective_bs

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _malloc_trim()

        logger.info(f"[phase] Effective batch size: {self._effective_batch_size}")

        self._cache = self._init_cache(max_cache_len, effective_bs)
        num_truncated = 0

        step = 0
        step_time_sum = 0.0
        early_promoted = 0

        while slot_queue:
            # --- Take a chunk of up to effective_bs slots ---
            chunk_slots: list[_Slot] = []
            self._valid_lens = [0] * effective_bs

            restore_start = time.perf_counter()
            while slot_queue and len(chunk_slots) < effective_bs:
                if not is_last and slot_queue[0].valid_len >= (total_threshold - self._PHASE_STEP * 0.1):
                    staged = slot_queue.popleft()
                    promote_queue.append(staged)
                    early_promoted += 1
                    continue

                staged = slot_queue.popleft()
                batch_idx = len(chunk_slots)
                keys, vals = kv_store.restore(staged)
                self._cache.restore_row_from_cpu(batch_idx, staged.valid_len, keys, vals)
                self._valid_lens[batch_idx] = staged.valid_len
                chunk_slots.append(staged.slot)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            restore_time = time.perf_counter() - restore_start
            logger.info(
                f"[perf] Restored {len(chunk_slots)} slots from CPU in {restore_time:.4f}s "
                f"ram_kv={kv_store._ram_bytes / 1e9:.2f}GB"
            )

            if early_promoted > 0:
                logger.info(f"[phase] Early promoted {early_promoted} slots that exceeded the phase budget")

            if not chunk_slots:
                continue

            # Compute batch-level phase budget
            max_valid_len = max(self._valid_lens[i] for i in range(len(chunk_slots)))
            phase_budget = total_threshold - max_valid_len

            finished: set[int] = set()
            chunk_step = 0

            # --- Decode until all slots in chunk are done or budget exhausted ---
            while len(finished) < len(chunk_slots):
                active = [(i, s) for i, s in enumerate(chunk_slots) if i not in finished]
                if not active:
                    break

                active_slots = [s for _, s in active]
                active_indices = [i for i, _ in active]
                max_active_len = max(self._valid_lens[i] for i in active_indices)

                step_start = time.perf_counter()
                self._batched_decode(active_indices, active_slots)
                step += 1
                chunk_step += 1

                # Sync only on log boundaries to avoid per-step GPU→CPU stalls.
                # Keeps the GPU pipeline full between syncs.
                if step % 200 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    step_time_sum += time.perf_counter() - step_start
                    avg_step = step_time_sum / step
                    rss_gb = psutil.Process().memory_info().rss / 1e9
                    logger.info(
                        f"[perf] phase step={step} active={len(active)} "
                        f"max_active_len={max_active_len} avg_step={avg_step:.4f}s "
                        f"queue={len(slot_queue)} rss={rss_gb:.2f}GB "
                        f"ram_kv={kv_store._ram_bytes / 1e9:.2f}GB"
                    )

                # Periodic safety net: abort phase if free VRAM is critically low
                if torch.cuda.is_available() and step % self._vram_check_interval == 0:
                    if self._check_vram_pressure():
                        self._vram_reduced_bs = max(effective_bs // 2, 1)
                        logger.info(
                            f"[vram] Free VRAM critically low at step {step}. Aborting phase. "
                            f"Reducing batch size for next attempt: {effective_bs} -> {self._vram_reduced_bs}."
                        )
                        # Stage active slots back off the GPU and re-queue
                        for i, slot in enumerate(chunk_slots):
                            if i not in finished:
                                valid_len = self._valid_lens[i]
                                keys, vals = self._cache.stage_row_to_cpu(i, valid_len)
                                promote_queue.appendleft(kv_store.stage(slot, valid_len, keys, vals))
                        while slot_queue:
                            promote_queue.append(slot_queue.popleft())
                        break

                # Check per-slot completion (EOS / max_new_tokens)
                for batch_idx, slot in active:
                    last_token = slot.generated_ids[-1]
                    if last_token == self.eos_token_id or len(slot.generated_ids) >= self.max_new_tokens:
                        if last_token != self.eos_token_id:
                            num_truncated += 1
                            truncated_flags[slot.index] = True
                        results[slot.index] = slot.generated_ids
                        thinking_exhausted_flags[slot.index] = slot.thinking_budget_exhausted
                        finished.add(batch_idx)
                        pbar.update(1)

                # Batch-level promotion: budget exhausted → stage all remaining off the GPU
                if chunk_step >= phase_budget:
                    for batch_idx, slot in enumerate(chunk_slots):
                        if batch_idx not in finished:
                            valid_len = self._valid_lens[batch_idx]
                            keys, vals = self._cache.stage_row_to_cpu(batch_idx, valid_len)
                            promote_queue.append(kv_store.stage(slot, valid_len, keys, vals))
                            finished.add(batch_idx)
                    break

        if step > 0:
            logger.info(f"[perf] Phase done: {step} decode steps, avg_step={step_time_sum / step:.4f}s")

        return num_truncated

    def _prefill(
        self,
        slot: _Slot,
        prefill_ids: list[int],
        batch_idx: int,
        resume_generated_ids: list[int] | None = None,
    ) -> None:
        """Run the prefill forward pass to build KV cache and sample the first token.

        Updates slot.generated_ids in-place with the newly sampled token.

        Resume mode (resume_generated_ids set): `prefill_ids` is prompt + all but
        the last generated token, so the KV is rebuilt for exactly the positions a
        non-interrupted run would have cached after generating those tokens. No new
        token is sampled; slot.generated_ids is restored to the full list and the
        next decode step continues bit-identically from the last generated token
        (sampling is argmax/temperature=0 in these evals).

        Args:
            slot: The slot to populate (index and prompt_ids must already be set).
            prefill_ids: Token IDs to prefill.
            batch_idx: Which row in the batch cache to use.
            resume_generated_ids: Full generated-token list when resuming, else None.
        """
        device = self.model.device
        input_ids = torch.tensor([prefill_ids], device=device)
        seq_len = len(prefill_ids)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        cache_position = torch.arange(seq_len, device=device)

        prefill_cache = self._cache.prefill_view(batch_idx, seq_len)

        outputs = self._uncompiled_model(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=prefill_cache,
            use_cache=True,
        )
        self._valid_lens[batch_idx] = seq_len

        if resume_generated_ids is None:
            next_token = self._sample_token(outputs.logits[:, -1, :])
            slot.generated_ids = [next_token.item()]
        else:
            slot.generated_ids = list(resume_generated_ids)
        # seq_len = prompt_len (fresh) or prompt_len + len(generated) - 1 (resume),
        # so seq_position = prompt_len + len(generated) in both cases.
        slot.seq_position = seq_len + 1

    def _batched_decode(self, batch_indices: list[int], slots: list[_Slot]) -> None:
        """Run a single batched decode step for all active slots.

        Each slot writes KV to its own cache position (no shared write + compact).
        Uses the pre-allocated cache — zero tensor allocations per step.
        Tensor batch dimension is sized to effective_batch_size (set per phase).

        Args:
            batch_indices: The batch row index for each slot.
            slots: The active slots to decode (parallel to batch_indices).
        """
        device = self.model.device

        # max_active_len determines attention mask width and cache view size.
        max_active_len = max(self._valid_lens[i] for i in batch_indices)
        self._cache._active_seq_len = max_active_len

        bs = self._effective_batch_size

        # --- Vectorised tensor construction (replaces 4 Python for-loops) ---
        # Gather Python-side data into CPU lists, then move to GPU in one shot.
        batch_idx_t = torch.tensor(batch_indices, device=device)
        valid_lens_t = torch.tensor([self._valid_lens[i] for i in batch_indices], device=device)
        last_tokens_t = torch.tensor([slot.generated_ids[-1] for slot in slots], dtype=torch.long, device=device)
        seq_positions_t = torch.tensor([slot.seq_position for slot in slots], dtype=torch.long, device=device)

        # per_row_positions: 1 scatter instead of ~94 element writes
        per_row_positions = torch.zeros(bs, dtype=torch.long, device=device)
        per_row_positions[batch_idx_t] = valid_lens_t
        self._cache._per_row_cache_positions = per_row_positions

        # input_ids [bs, 1]: 1 scatter instead of ~94 element writes
        input_ids = torch.full((bs, 1), self.pad_token_id, dtype=torch.long, device=device)
        input_ids[batch_idx_t, 0] = last_tokens_t

        # attention_mask [bs, seq]: vectorised broadcast instead of ~188 element writes
        seq_width = max_active_len + 1
        cols = torch.arange(seq_width, device=device).unsqueeze(0)  # (1, seq)
        attn_mask = torch.zeros(bs, seq_width, dtype=torch.long, device=device)
        attn_mask[batch_idx_t] = (cols <= valid_lens_t.unsqueeze(1)).long()

        # cache_position: max_active_len for correct causal mask sizing
        cache_position = torch.tensor([max_active_len], device=device)

        # position_ids [bs, 1]: 1 scatter instead of ~94 element writes
        position_ids = torch.zeros(bs, 1, dtype=torch.long, device=device)
        position_ids[batch_idx_t, 0] = seq_positions_t

        # Tell the unpad cache the max seq length so it can skip .item() sync
        self._current_max_seqlen = seq_width

        # Without these hints, every distinct (bs, seq_width) combination
        # triggers a fresh inductor compile. Across phases and chunks that
        # is thousands of compiles — the worker pool eventually deadlocks
        # (subproc_pool._recv_msg stuck) and the cgroup OOMKills.
        torch._dynamo.mark_dynamic(input_ids, 0)
        torch._dynamo.mark_dynamic(attn_mask, 0)
        torch._dynamo.mark_dynamic(attn_mask, 1)
        torch._dynamo.mark_dynamic(position_ids, 0)
        torch._dynamo.mark_dynamic(cache_position, 0)

        # Single forward() call — model writes new KV via per-row cache positions
        outputs = self._compiled_model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=self._cache,
            use_cache=True,
        )

        # Clear the per-forward unpad cache so next step recomputes
        self._unpad_clear()

        # Sample next token per slot and advance valid_lens
        for i, slot in zip(batch_indices, slots):
            next_token = self._sample_token(outputs.logits[i : i + 1, -1, :])
            slot.generated_ids.append(next_token.item())
            if self._enforce_thinking_cap and slot.in_thinking:
                if slot.generated_ids[-1] == self.thinking_end_token_id:
                    slot.in_thinking = False
                elif len(slot.generated_ids) >= self.max_thinking_tokens:
                    # Force end-of-thinking. Replace the just-sampled token with
                    # </think> so total budget stays at max_new_tokens. KV at this
                    # position reflects the originally sampled token; next forward
                    # pass attends to a 1-token-stale K/V row. Acceptable for
                    # research-mode forced decoding (cf. speculative decoding).
                    slot.generated_ids[-1] = self.thinking_end_token_id
                    slot.in_thinking = False
                    slot.thinking_budget_exhausted = True
            slot.seq_position += 1
            self._valid_lens[i] += 1

    def _sample_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample a single token from logits of shape [1, vocab_size]."""
        if self.temperature == 0.0:
            return logits.argmax(dim=-1).squeeze(0)

        logits = logits / self.temperature

        if self.top_k > 0:
            top_k = min(self.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1).squeeze(0)
