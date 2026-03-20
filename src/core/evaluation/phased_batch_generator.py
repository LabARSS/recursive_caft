import gc
import time
import types
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch
from tqdm import tqdm
from transformers import DynamicCache, PreTrainedModel, PreTrainedTokenizer
from transformers.cache_utils import _static_cache_update


@dataclass
class GenerationResult:
    sequences: list[list[int]]
    num_truncated: int
    total: int


@dataclass
class _Slot:
    index: int
    prompt_ids: list[int]
    prompt_len: int
    generated_ids: list[int] = field(default_factory=list)
    seq_position: int = 0  # total tokens seen = prompt_len + len(generated_ids)


@dataclass
class _StagedSlot:
    """Slot with KV cache staged on CPU."""

    slot: _Slot
    valid_len: int
    # Per-layer CPU tensors: shape [1, num_kv_heads, valid_len, head_dim]
    key_cache: list[torch.Tensor]
    value_cache: list[torch.Tensor]


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

        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self._effective_batch_size = max_batch_size
        self._cache = None
        self._vram_safe_margin = 0.10  # reserve 10% of total VRAM as buffer
        self._vram_check_interval = 50  # periodic free-VRAM safety check every N decode steps
        self._vram_reduced_bs: int | None = None  # sticky reduced bs after VRAM pressure

        self._patch_causal_mask(self._uncompiled_model)

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
                if attention_mask is not None and torch.any(attention_mask == 0):
                    return attention_mask
                return None
            return original(attention_mask, input_tensor, cache_position, past_key_values, output_attentions)

        candidate._update_causal_mask = types.MethodType(_update_causal_mask, candidate)

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

    _PHASE_STEP = 1024  # max tokens generated per phase

    @torch.no_grad()
    def generate(self, prompts: list[list[int]]) -> GenerationResult:
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
        pbar = tqdm(total=len(prompts), desc="Generating")
        max_prompt_len = max(len(p) for p in prompts)
        max_total = max_prompt_len + self.max_new_tokens

        # --- Prefill all prompts upfront (batch_size=1 temp cache) ---
        prefill_cache = self._init_cache(max_prompt_len + 1, 1)
        slot_queue: deque[_StagedSlot] = deque()

        prefill_start = time.perf_counter()
        total_prefill_tokens = 0
        for i, prompt_ids in enumerate(tqdm(prompts, desc="Prefilling", leave=False)):
            slot = _Slot(index=i, prompt_ids=prompt_ids, prompt_len=len(prompt_ids))
            self._cache = prefill_cache
            self._valid_lens = [0]
            self._prefill(slot, prompt_ids, 0)
            total_prefill_tokens += len(prompt_ids)

            # Stage KV to CPU
            valid_len = self._valid_lens[0]
            keys, vals = prefill_cache.stage_row_to_cpu(0, valid_len)
            slot_queue.append(_StagedSlot(slot=slot, valid_len=valid_len, key_cache=keys, value_cache=vals))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prefill_time = time.perf_counter() - prefill_start
        pbar.write(
            f"[perf] Prefill: {len(prompts)} prompts, {total_prefill_tokens} tokens, "
            f"{prefill_time:.4f}s ({total_prefill_tokens / prefill_time:.0f} tok/s)"
        )

        # Free prefill cache
        self._cache = None
        prefill_cache = None

        # --- Phase loop ---
        pbar.write(
            f"[phase] Starting generation: {len(slot_queue)} prompts, "
            f"phase_step={self._PHASE_STEP} max={self.max_new_tokens}"
        )

        trunc = 0
        phase = 0
        total_threshold = min(self._PHASE_STEP, max_total)

        while True:
            promote_queue: deque[_StagedSlot] = deque()

            pbar.write(
                f"[phase] Starting phase {phase + 1}: {len(slot_queue)} sequences, total_threshold={total_threshold}"
            )

            is_last = total_threshold >= max_total
            trunc += self._run_phase(
                slot_queue, results, total_threshold, promote_queue, pbar, total_threshold + 1, is_last
            )

            self._cache = None

            if not promote_queue:
                break

            slot_queue = promote_queue
            phase += 1
            total_threshold = min(self._PHASE_STEP * (phase + 1), max_total)
            pbar.write(f"[phase] Phase {phase}: {len(slot_queue)} sequences promoted")

        pbar.close()
        sequences = [r if r is not None else [] for r in results]
        return GenerationResult(sequences=sequences, num_truncated=trunc, total=len(prompts))

    def _run_phase(
        self,
        slot_queue: deque[_StagedSlot],
        results: list[list[int] | None],
        total_threshold: int,
        promote_queue: deque[_StagedSlot],
        pbar: tqdm,
        max_cache_len: int,
        is_last: bool,
    ) -> int:
        """Run a complete phase with static batching.

        Pops staged slots from slot_queue, restores KV from CPU → GPU,
        decodes until done or budget exhausted, stages promoted slots back
        to CPU.

        Returns the number of truncated sequences.
        """
        self._cache = None

        max_bs = self._vram_reduced_bs if self._vram_reduced_bs is not None else self.max_batch_size
        effective_bs = min(len(slot_queue), max_bs)
        if effective_bs != self._effective_batch_size:
            pbar.write(
                f"[perf] Adjusting batch size: {self._effective_batch_size} → {effective_bs}. torch.compile will take time."
            )
            self._effective_batch_size = effective_bs

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pbar.write(f"[phase] Effective batch size: {self._effective_batch_size}")

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
                if not is_last and (slot_queue[0].valid_len - self._PHASE_STEP * 0.1) >= total_threshold:
                    staged = slot_queue.popleft()
                    promote_queue.append(staged)
                    early_promoted += 1
                    continue

                staged = slot_queue.popleft()
                batch_idx = len(chunk_slots)
                self._cache.restore_row_from_cpu(batch_idx, staged.valid_len, staged.key_cache, staged.value_cache)
                self._valid_lens[batch_idx] = staged.valid_len
                chunk_slots.append(staged.slot)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            restore_time = time.perf_counter() - restore_start
            pbar.write(f"[perf] Restored {len(chunk_slots)} slots from CPU in {restore_time:.4f}s")

            if early_promoted > 0:
                pbar.write(f"[phase] Early promoted {early_promoted} slots that exceeded the phase budget")

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
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_time_sum += time.perf_counter() - step_start

                if step % 200 == 0:
                    avg_step = step_time_sum / (step + 1)
                    pbar.write(
                        f"[perf] phase step={step} active={len(active)} "
                        f"max_active_len={max_active_len} avg_step={avg_step:.4f}s "
                        f"queue={len(slot_queue)}"
                    )
                step += 1
                chunk_step += 1

                # Periodic safety net: abort phase if free VRAM is critically low
                if torch.cuda.is_available() and step % self._vram_check_interval == 0:
                    if self._check_vram_pressure():
                        self._vram_reduced_bs = max(effective_bs // 2, 1)
                        pbar.write(
                            f"[vram] Free VRAM critically low at step {step}. Aborting phase. "
                            f"Reducing batch size for next attempt: {effective_bs} -> {self._vram_reduced_bs}."
                        )
                        # Stage active slots back to CPU and re-queue
                        for i, slot in enumerate(chunk_slots):
                            if i not in finished:
                                valid_len = self._valid_lens[i]
                                keys, vals = self._cache.stage_row_to_cpu(i, valid_len)
                                promote_queue.appendleft(
                                    _StagedSlot(
                                        slot=slot,
                                        valid_len=valid_len,
                                        key_cache=keys,
                                        value_cache=vals,
                                    )
                                )
                        break

                # Check per-slot completion (EOS / max_new_tokens)
                for batch_idx, slot in active:
                    last_token = slot.generated_ids[-1]
                    if last_token == self.eos_token_id or len(slot.generated_ids) >= self.max_new_tokens:
                        if last_token != self.eos_token_id:
                            num_truncated += 1
                        results[slot.index] = slot.generated_ids
                        finished.add(batch_idx)
                        pbar.update(1)

                # Batch-level promotion: budget exhausted → stage all remaining to CPU
                if chunk_step >= phase_budget:
                    for batch_idx, slot in enumerate(chunk_slots):
                        if batch_idx not in finished:
                            valid_len = self._valid_lens[batch_idx]
                            keys, vals = self._cache.stage_row_to_cpu(batch_idx, valid_len)
                            promote_queue.append(
                                _StagedSlot(
                                    slot=slot,
                                    valid_len=valid_len,
                                    key_cache=keys,
                                    value_cache=vals,
                                )
                            )
                            finished.add(batch_idx)
                    break

        if step > 0:
            pbar.write(f"[perf] Phase done: {step} decode steps, avg_step={step_time_sum / step:.4f}s")

        return num_truncated

    def _prefill(
        self,
        slot: _Slot,
        prefill_ids: list[int],
        batch_idx: int,
    ) -> None:
        """Run the prefill forward pass to build KV cache and sample the first token.

        Updates slot.generated_ids in-place with the newly sampled token.

        Args:
            slot: The slot to populate (index and prompt_ids must already be set).
            prefill_ids: Token IDs to prefill.
            batch_idx: Which row in the batch cache to use.
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

        next_token = self._sample_token(outputs.logits[:, -1, :])
        slot.generated_ids = [next_token.item()]
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

        # Set per-row cache positions so each slot writes KV at its own valid_len.
        bs = self._effective_batch_size
        per_row_positions = torch.zeros(bs, dtype=torch.long, device=device)
        for i in batch_indices:
            per_row_positions[i] = self._valid_lens[i]
        self._cache._per_row_cache_positions = per_row_positions

        # Build input_ids [bs, 1]
        input_ids = torch.full((bs, 1), self.pad_token_id, dtype=torch.long, device=device)
        for i, slot in zip(batch_indices, slots):
            input_ids[i, 0] = slot.generated_ids[-1]

        # Build attention_mask [bs, max_active_len + 1]
        attn_mask = torch.zeros(bs, max_active_len + 1, dtype=torch.long, device=device)
        for i in batch_indices:
            valid_len = self._valid_lens[i]
            attn_mask[i, :valid_len] = 1  # valid cached positions
            attn_mask[i, valid_len] = 1  # new token position

        # cache_position: max_active_len for correct causal mask sizing
        cache_position = torch.tensor([max_active_len], device=device)

        # position_ids [bs, 1]
        position_ids = torch.zeros(bs, 1, dtype=torch.long, device=device)
        for i, slot in zip(batch_indices, slots):
            position_ids[i, 0] = slot.seq_position

        # Single forward() call — model writes new KV via per-row cache positions
        outputs = self._compiled_model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=self._cache,
            use_cache=True,
        )

        # Sample next token per slot and advance valid_lens
        for i, slot in zip(batch_indices, slots):
            next_token = self._sample_token(outputs.logits[i : i + 1, -1, :])
            slot.generated_ids.append(next_token.item())
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
