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


# Set once if a pinned host allocation fails (e.g. cudaHostAlloc OOM / a low
# RLIMIT_MEMLOCK in a container): we stop attempting to pin for the rest of the
# process and fall back to pageable RAM.
_PINNING_DISABLED = False


def _to_pinned_host(stacked_gpu: torch.Tensor, pin: bool) -> torch.Tensor:
    """Blocking D2H copy of a stacked KV tensor into host memory.

    When `pin` is set (and pinning hasn't been disabled by a prior failure), the
    destination is page-locked, so the copy uses the DMA fast path and a later
    host->device restore can run async. The copy is intentionally *blocking*:
    pinning — not non_blocking — is what speeds the transfer, and a blocking copy
    keeps the transient GPU stack alive until it completes. Falls back to a plain
    pageable copy on allocation failure (logged once).
    """
    global _PINNING_DISABLED
    if pin and not _PINNING_DISABLED and torch.cuda.is_available():
        try:
            dest = torch.empty(stacked_gpu.shape, dtype=stacked_gpu.dtype, pin_memory=True)
            dest.copy_(stacked_gpu)
            return dest
        except RuntimeError as exc:
            _PINNING_DISABLED = True
            logger.warning(
                f"[kv-store] pinned host allocation failed ({exc}); "
                "falling back to pageable KV staging for the rest of this run"
            )
    return stacked_gpu.to("cpu")


@dataclass
class GenerationResult:
    sequences: list[list[int]]
    num_truncated: int
    total: int
    truncated: list[bool] = field(default_factory=list)
    thinking_budget_exhausted: list[bool] = field(default_factory=list)
    # True for sequences stopped at a per-call carry boundary (length-stage cap)
    # without hitting EOS or the global max_new_tokens: still-running survivors the
    # caller pools across chunks and re-prefills in the next stage. Always all-False
    # when generate() is called without carry_at_new_tokens (the default).
    unfinished: list[bool] = field(default_factory=list)


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

    Either RAM-resident (keys/vals set, spill_path None) or disk-resident
    (spill_path set, keys/vals None). Created and consumed exclusively through
    _StagedKVStore.
    """

    slot: _Slot
    valid_len: int
    nbytes: int  # total CPU bytes of this slot's K+V tensors
    # One contiguous CPU tensor each, shape [num_layers, 1, num_kv_heads,
    # valid_len, head_dim]. None when the slot has been spilled to disk.
    keys: torch.Tensor | None = None
    vals: torch.Tensor | None = None
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

    Each slot's K and V are already single contiguous tensors (stacked across
    layers at stage time), so disk spill is just a torch.save of those two
    tensors — one file per slot.
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
    def _kv_nbytes(keys: torch.Tensor, vals: torch.Tensor) -> int:
        """Total CPU bytes of a slot's stacked K and V tensors."""
        return keys.numel() * keys.element_size() + vals.numel() * vals.element_size()

    def stage(
        self,
        slot: _Slot,
        valid_len: int,
        keys: torch.Tensor,
        vals: torch.Tensor,
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
        return _StagedSlot(slot=slot, valid_len=valid_len, nbytes=nbytes, keys=keys, vals=vals)

    def restore(self, staged: _StagedSlot) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (keys, vals) stacked CPU tensors; read+delete the file for disk slots."""
        if staged.spill_path is not None:
            keys, vals = self._read_from_disk(staged.spill_path)
            os.remove(staged.spill_path)
            return keys, vals
        self._ram_bytes -= staged.nbytes
        keys, vals = staged.keys, staged.vals
        staged.keys = staged.vals = None
        return keys, vals

    def _spill_to_disk(self, keys: torch.Tensor, vals: torch.Tensor) -> str:
        """Write a slot's stacked K and V tensors to one file. Returns the path."""
        path = os.path.join(self.spill_dir, f"kv_{self._spill_seq:08d}.pt")
        self._spill_seq += 1
        try:
            torch.save({"k": keys, "v": vals}, path)
        except Exception:
            # torch.save's zip-write errors (`unexpected pos X vs Y`,
            # `basic_ios::clear: iostream error`) almost always mean disk-full
            # or I/O fault. Surface enough state to tell which without needing
            # the formatter's locals renderer (which itself fails on tensors).
            free_bytes = shutil.disk_usage(self.spill_dir).free
            nbytes = keys.numel() * keys.element_size() + vals.numel() * vals.element_size()
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
    def _read_from_disk(path: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Load a spilled slot's stacked K and V tensors."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return payload["k"], payload["v"]

    def close(self) -> None:
        """Delete the spill directory. Safe to call multiple times."""
        logger.info(
            f"[kv-store] close spilled_slots={self._spilled_count} spilled_bytes={self._spilled_bytes / 1e9:.2f}GB"
        )
        shutil.rmtree(self.spill_dir, ignore_errors=True)
        _malloc_trim()


class _PreAllocatedBatchCache(DynamicCache):
    """Pre-allocated KV cache for batched decode with per-row positions.

    update() does a branchless per-row indexed write, so rows at different
    sequence lengths can share one decode step. Prefill uses a plain
    DynamicCache (see BatchGenerator._prefill).
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_batch_size: int,
        max_cache_len: int,
        device: torch.device,
        dtype: torch.dtype,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.max_cache_len = max_cache_len
        self._pin_memory = pin_memory
        self._active_seq_len = 0
        self._per_row_cache_positions = torch.zeros(max_batch_size, dtype=torch.long, device=device)
        self._batch_indices = torch.arange(max_batch_size, device=device)
        cache_shape = (max_batch_size, num_kv_heads, max_cache_len, head_dim)
        for _ in range(num_layers):
            k = torch.zeros(cache_shape, dtype=dtype, device=device)
            v = torch.zeros(cache_shape, dtype=dtype, device=device)
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

    def stage_row_to_cpu(self, batch_idx: int, valid_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Copy one batch row's KV into a single contiguous CPU tensor each.

        Returns (keys, vals) of shape [num_layers, 1, kv_heads, valid_len, head_dim].
        One GPU stack + one device->host copy per K/V (vs one tiny copy per layer),
        so each slot costs 2 host allocations instead of 2*num_layers. Host tensors
        are page-locked when pin_memory is set (fast DMA + async restore later).
        """
        keys_gpu = torch.stack(
            [self.key_cache[i][batch_idx : batch_idx + 1, :, :valid_len, :] for i in range(len(self))], dim=0
        )
        vals_gpu = torch.stack(
            [self.value_cache[i][batch_idx : batch_idx + 1, :, :valid_len, :] for i in range(len(self))], dim=0
        )
        return _to_pinned_host(keys_gpu, self._pin_memory), _to_pinned_host(vals_gpu, self._pin_memory)

    def restore_row_from_cpu(self, batch_idx: int, valid_len: int, keys: torch.Tensor, vals: torch.Tensor) -> None:
        """Copy a slot's stacked CPU KV back into a batch row on GPU.

        keys/vals: [num_layers, 1, kv_heads, valid_len, head_dim]. One host->device
        copy per K/V, then per-layer GPU->GPU slice-assign (no host allocations).
        The H2D copy is async only for pinned sources (safe — the caller syncs before
        decode); pageable sources (disk-restored slots) fall back to a blocking copy.
        """
        k_gpu = keys.to(self.key_cache[0].device, non_blocking=keys.is_pinned())
        v_gpu = vals.to(self.value_cache[0].device, non_blocking=vals.is_pinned())
        for i in range(len(self)):
            self.key_cache[i][batch_idx, :, :valid_len, :] = k_gpu[i, 0]
            self.value_cache[i][batch_idx, :, :valid_len, :] = v_gpu[i, 0]


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
        kv_cache_offload_threshold_gb: float = 180.0,
        kv_cache_spill_dir: str | None = None,
        kv_cache_pin_memory: bool = True,
    ):
        self.model = model
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
        self._kv_cache_pin_memory = kv_cache_pin_memory
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

        self._patch_causal_mask(self.model)
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
                # redundant and caused a GPU→CPU sync.
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
            pin_memory=self._kv_cache_pin_memory,
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

    _PHASE_STEP = 512  # max tokens generated per phase

    @torch.no_grad()
    def generate(
        self,
        prompts: list[list[int]],
        checkpoint_path: str | None = None,
        carry_at_new_tokens: int | None = None,
        resume_generated: list[list[int] | None] | None = None,
        resume_thinking_exhausted: list[bool] | None = None,
    ) -> GenerationResult:
        """Generate responses for a list of prompts using phased generation.

        All prompts are prefilled upfront (one-by-one, batch_size=1) with KV
        cache staged on CPU. Then decode phases restore slots to GPU, decode
        in batches, and re-stage promoted slots to CPU for the next phase.

        Length-stage carry (optional): when `carry_at_new_tokens` is set, a
        sequence that reaches that many generated tokens without hitting EOS or
        the global `max_new_tokens` is stopped and returned as *unfinished*
        (GenerationResult.unfinished[i] == True) rather than truncated. The caller
        pools these survivors across chunks and re-issues them in the next stage
        via `resume_generated`. With `carry_at_new_tokens=None` (the default)
        behavior is identical to single-pass generation.

        Args:
            prompts: List of token ID sequences (one per sample).
            checkpoint_path: Per-call crash-resume checkpoint (token ids only).
            carry_at_new_tokens: Stop sequences at this many generated tokens and
                mark them unfinished (None → run to EOS/max_new_tokens as before).
            resume_generated: Per-prompt tokens already generated in a prior stage
                (None entry → fresh prompt). Rebuilt via the prefill resume path.
            resume_thinking_exhausted: Per-prompt thinking-budget-exhausted state to
                restore for seeded rows (parallel to `resume_generated`).

        Returns:
            GenerationResult with generated sequences and truncation/unfinished stats.
        """
        results: list[list[int] | None] = [None] * len(prompts)
        truncated_flags: list[bool] = [False] * len(prompts)
        thinking_exhausted_flags: list[bool] = [False] * len(prompts)
        unfinished_flags: list[bool] = [False] * len(prompts)
        pbar = tqdm(total=len(prompts), desc="Generating")
        max_prompt_len = max(len(p) for p in prompts)
        # When a carry cap is set, the phase loop runs only to that many new tokens
        # and returns still-running sequences as unfinished. The cap (clamped to
        # max_new_tokens by the caller) drives max_total so total_threshold /
        # is_last / phase_budget / cache sizing all stop at the stage boundary.
        effective_budget = carry_at_new_tokens if carry_at_new_tokens is not None else self.max_new_tokens
        max_total = max_prompt_len + effective_budget

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
            seed = resume_generated[i] if resume_generated is not None else None
            if row is not None and row["done"]:
                results[i] = row["generated_ids"]
                truncated_flags[i] = row["truncated"]
                thinking_exhausted_flags[i] = row["thinking_budget_exhausted"]
                pbar.update(1)
            elif row is not None and row["generated_ids"]:
                # In-flight checkpoint (strictly newer than any stage seed): rebuild
                # KV for prompt + all-but-last generated token, then resume decoding
                # from the last generated token (see _prefill).
                gen = row["generated_ids"]
                prefill_plan.append((i, prompt_ids + gen[:-1], gen))
            elif seed:
                # Cross-stage carry seed: this prompt already generated `seed` tokens
                # in a prior length-stage. Same re-prefill path as checkpoint resume.
                prefill_plan.append((i, prompt_ids + seed[:-1], seed))
            else:
                prefill_plan.append((i, prompt_ids, None))

        if ckpt is not None:
            logger.info(
                f"[ckpt] resuming {checkpoint_path}: completed_phase={resume_phase}, "
                f"{sum(1 for r in results if r is not None)} done, {len(prefill_plan)} to continue"
            )

        # --- (Re-)prefill the unfinished prompts (each in its own DynamicCache) ---
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
                keys, vals, valid_len = self._prefill(slot, prefill_ids, resume_generated_ids=gen)
                total_prefill_tokens += len(prefill_ids)
                if gen is not None and self._enforce_thinking_cap:
                    # Reconstruct thinking-cap state from the generated tokens.
                    slot.in_thinking = self.thinking_end_token_id not in gen
                    ckpt_row = resume_rows.get(str(i))
                    if ckpt_row is not None:
                        slot.thinking_budget_exhausted = ckpt_row["thinking_budget_exhausted"]
                    elif resume_thinking_exhausted is not None:
                        slot.thinking_budget_exhausted = resume_thinking_exhausted[i]

                # Stage KV off the GPU (RAM, or disk once over threshold)
                slot_queue.append(kv_store.stage(slot, valid_len, keys, vals))

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            prefill_time = time.perf_counter() - prefill_start
            logger.info(
                f"[perf] Prefill: {len(prefill_plan)} prompts, {total_prefill_tokens} tokens, "
                f"{prefill_time:.4f}s ({total_prefill_tokens / max(prefill_time, 1e-9):.0f} tok/s)"
            )

            self._measure_usable_vram()

            # --- Phase loop ---
            logger.info(
                f"[phase] Starting generation: {len(slot_queue)} prompts, "
                f"phase_step={self._PHASE_STEP} max={self.max_new_tokens}"
            )

            # Start the phase counter so total_threshold immediately clears the
            # already-staged length. Fresh prompts start at 0; seeded survivors
            # (stage > 0) start past their prior tokens so we don't churn empty
            # phases early-promoting every slot until the threshold catches up.
            has_seed = resume_generated is not None and any(s for s in resume_generated)
            if resume_phase is not None:
                phase = resume_phase + 1
            elif has_seed and slot_queue:
                phase = max(s.valid_len for s in slot_queue) // self._PHASE_STEP
            else:
                phase = 0
            total_threshold = min(self._PHASE_STEP * (phase + 1), max_total)

            # Bound before the loop so the post-loop carry sweep is safe even when the
            # loop breaks on the first iteration (e.g. a fully-resumed sub-chunk).
            promote_queue: deque[_StagedSlot] = deque()
            while True:
                if not slot_queue:
                    break  # nothing left to generate (e.g. a fully-resumed chunk)
                promote_queue = deque()

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
                    unfinished_flags,
                    total_threshold,
                    promote_queue,
                    pbar,
                    total_threshold + 1,
                    is_last,
                    carry_at_new_tokens,
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
                # Carried survivors are written as in-flight (done=False) so a
                # resume re-decodes and re-carries them idempotently.
                self._write_checkpoint(
                    checkpoint_path,
                    phase,
                    results,
                    truncated_flags,
                    thinking_exhausted_flags,
                    unfinished_flags,
                    promote_queue,
                )

                if not promote_queue:
                    break
                # With a carry cap, a non-empty promote_queue on the final phase
                # (is_last) is carried survivors — e.g. staged off the GPU by a
                # VRAM-pressure abort. Stop and record them as unfinished (below)
                # instead of re-running. Without a carry cap, fall through to the
                # normal re-run path (preserves VRAM-abort recovery on the last phase).
                if carry_at_new_tokens is not None and is_last:
                    break

                slot_queue = promote_queue
                phase += 1
                total_threshold = min(self._PHASE_STEP * (phase + 1), max_total)
                logger.info(f"[phase] Phase {phase}: {len(slot_queue)} sequences promoted")

            # Record any sequence still queued at the carry boundary as an unfinished
            # survivor. Normally empty (hit_carry already recorded carries directly);
            # this only catches slots a VRAM abort staged off the GPU on the last phase.
            if carry_at_new_tokens is not None:
                for staged in promote_queue:
                    s = staged.slot
                    if results[s.index] is None:
                        results[s.index] = s.generated_ids
                        thinking_exhausted_flags[s.index] = s.thinking_budget_exhausted
                        unfinished_flags[s.index] = True
                        pbar.update(1)
                promote_queue.clear()

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
                unfinished=unfinished_flags,
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
        unfinished_flags: list[bool],
        promote_queue: "deque[_StagedSlot]",
    ) -> None:
        """Persist finished results + in-flight token ids after a phase (no KV).

        Token ids only (a few MB), written atomically (tmp + os.replace) once per
        phase — negligible next to the multi-second per-phase KV restores. Lets a
        re-launched process resume this chunk via generate(checkpoint_path=...).

        Carried survivors (a carry boundary stopped them, `unfinished_flags[i]`)
        are persisted as in-flight (done=False) so resume re-decodes and re-carries
        them idempotently rather than treating them as finished.
        """
        if not checkpoint_path:
            return
        rows: dict[str, dict] = {}
        for i, gen in enumerate(results):
            if gen is not None:
                rows[str(i)] = {
                    "generated_ids": gen,
                    "done": not unfinished_flags[i],
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
        unfinished_flags: list[bool],
        total_threshold: int,
        promote_queue: deque[_StagedSlot],
        pbar: tqdm,
        max_cache_len: int,
        is_last: bool,
        carry_cap: int | None = None,
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

        # Track the slot count directly. With eager execution (no torch.compile),
        # phase-to-phase bs changes are free, so the prior hysteresis policy
        # (suppress changes <25%) only wasted compute on padding rows. Just snap
        # to even for warp-aligned kernels.
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
                        free, total = torch.cuda.mem_get_info()
                        logger.info(
                            f"[vram] Free VRAM critically low at step {step}: "
                            f"free={free / 1e9:.1f}GB/{total / 1e9:.1f}GB "
                            f"alloc={torch.cuda.memory_allocated() / 1e9:.1f}GB "
                            f"reserved={torch.cuda.memory_reserved() / 1e9:.1f}GB. Aborting phase. "
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

                # Check per-slot completion (EOS / max_new_tokens / carry boundary)
                for batch_idx, slot in active:
                    last_token = slot.generated_ids[-1]
                    n = len(slot.generated_ids)
                    hit_eos = last_token == self.eos_token_id
                    hit_global = n >= self.max_new_tokens
                    # Carry boundary (length-stage cap): stop without finishing so the
                    # caller pools this sequence across chunks and re-prefills it next
                    # stage. EOS and the global cap take precedence — a sequence that
                    # also ended naturally or hit the hard budget is genuinely done.
                    hit_carry = carry_cap is not None and n >= carry_cap and not hit_eos and not hit_global
                    if hit_eos or hit_global or hit_carry:
                        if hit_carry:
                            unfinished_flags[slot.index] = True
                        elif not hit_eos:
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
        resume_generated_ids: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Prefill one prompt (batch_size=1) and return its staged KV + length.

        Builds KV in a plain DynamicCache, samples the first token (unless
        resuming), and returns (keys, vals, valid_len) where keys/vals are
        contiguous CPU tensors of shape [num_layers, 1, kv_heads, seq_len,
        head_dim] — the same format _StagedKVStore.stage expects.

        Resume mode (resume_generated_ids set): `prefill_ids` is prompt + all but
        the last generated token, so the KV is rebuilt for exactly the positions a
        non-interrupted run would have cached after generating those tokens. No new
        token is sampled; slot.generated_ids is restored to the full list and the
        next decode step continues bit-identically from the last generated token
        (sampling is argmax/temperature=0 in these evals).

        Args:
            slot: The slot to populate (index and prompt_ids must already be set).
            prefill_ids: Token IDs to prefill.
            resume_generated_ids: Full generated-token list when resuming, else None.
        """
        device = self.model.device
        input_ids = torch.tensor([prefill_ids], device=device)
        seq_len = len(prefill_ids)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        cache_position = torch.arange(seq_len, device=device)

        cache = DynamicCache()
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
        )

        if resume_generated_ids is None:
            next_token = self._sample_token(outputs.logits[:, -1, :])
            slot.generated_ids = [next_token.item()]
        else:
            slot.generated_ids = list(resume_generated_ids)
        # seq_len = prompt_len (fresh) or prompt_len + len(generated) - 1 (resume),
        # so seq_position = prompt_len + len(generated) in both cases.
        slot.seq_position = seq_len + 1

        # Stage as two contiguous CPU tensors (one stack + one copy per K/V),
        # matching _PreAllocatedBatchCache.stage_row_to_cpu's format. Page-locked
        # when pin_memory is set, for a fast DMA restore later.
        pin = self._kv_cache_pin_memory
        keys_gpu = torch.stack([k[:, :, :seq_len, :] for k in cache.key_cache], dim=0)
        vals_gpu = torch.stack([v[:, :, :seq_len, :] for v in cache.value_cache], dim=0)
        return _to_pinned_host(keys_gpu, pin), _to_pinned_host(vals_gpu, pin), seq_len

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

        # Guard against an out-of-bounds KV-cache write — that would be a CUDA
        # illegal memory access that surfaces later (e.g. in a linear) as a
        # segfault. The write positions ARE the Python-side valid_lens, so this
        # is a CPU-only check with no GPU sync.
        max_pos = max(self._valid_lens[i] for i in batch_indices)
        assert max_pos < self._cache.max_cache_len, (
            f"KV-cache OOB write: pos={max_pos} >= max_cache_len={self._cache.max_cache_len} "
            f"(bs={bs}, active={len(batch_indices)}, max_active_len={max_active_len})"
        )

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

        # Single forward() call — model writes new KV via per-row cache positions
        outputs = self.model(
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
