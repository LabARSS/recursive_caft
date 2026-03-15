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
    batch_idx: int
    generated_ids: list[int] = field(default_factory=list)
    seq_position: int = 0  # total tokens seen = prompt_len + len(generated_ids)


@dataclass
class _PromotedSequence:
    index: int  # position in results array
    prompt_ids: list[int]  # original prompt tokens
    generated_ids: list[int]  # tokens generated so far


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


class ContinuousBatchGenerator:
    """Token-by-token generation with continuous batching via model.forward().

    Maintains a pool of active slots backed by a single pre-allocated KV cache.
    Empty slots are filled from a queue of pending prompts.

    Prefill runs individually per prompt (no padding waste). Decode batches
    all active slots into a single forward() call using an attention mask to
    ignore padded positions. The KV cache is updated in-place with zero
    tensor allocations per decode step.
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
        group_scheduling: bool | None = None,
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
        # Enable phase-based group scheduling. Default: auto (on when max_new_tokens >= 1000).
        if group_scheduling is None:
            self.group_scheduling = max_new_tokens >= 1000
        else:
            self.group_scheduling = group_scheduling

        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self._effective_batch_size = max_batch_size
        self._cache = None
        self._vram_safe_margin = 0.15  # reserve 15% of total VRAM as buffer
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

    def _force_requeue_active(
        self,
        active_slots: list[_Slot | None],
        input_queue: deque[tuple[int, list[int], list[int]]],
    ) -> None:
        """Push all active slots back to input_queue for retry with reduced batch size."""
        for i, slot in enumerate(active_slots):
            if slot is None:
                continue
            prefill_ids = slot.prompt_ids + slot.generated_ids
            input_queue.appendleft((slot.index, slot.prompt_ids, prefill_ids))
            active_slots[i] = None

    @torch.no_grad()
    def generate(self, prompts: list[list[int]]) -> GenerationResult:
        """Generate responses for a list of prompts using continuous batching.

        When group_scheduling is enabled, uses three sequential phases (fast, medium,
        slow) to keep max_active_len low. Otherwise uses a single flat phase.

        Args:
            prompts: List of token ID sequences (one per sample).

        Returns:
            GenerationResult with generated sequences and truncation stats.
        """
        results: list[list[int] | None] = [None] * len(prompts)
        num_truncated = 0
        pbar = tqdm(total=len(prompts), desc="Generating")
        max_prompt_len = max(len(p) for p in prompts)

        if not self.group_scheduling:
            # Simple single-phase path (loop handles VRAM-driven re-queue)
            input_queue: deque[tuple[int, list[int], list[int]]] = deque((i, p, p) for i, p in enumerate(prompts))
            cache_len = max_prompt_len + self.max_new_tokens
            while input_queue:
                num_truncated += self._run_phase(input_queue, results, self.max_new_tokens, None, pbar, cache_len)
        else:
            num_truncated = self._generate_grouped(prompts, results, pbar, max_prompt_len)

        pbar.close()
        sequences = [r if r is not None else [] for r in results]
        return GenerationResult(sequences=sequences, num_truncated=num_truncated, total=len(prompts))

    _PHASE_STEP = 1024  # max tokens generated per phase

    def _generate_grouped(
        self,
        prompts: list[list[int]],
        results: list[list[int] | None],
        pbar: tqdm,
        max_prompt_len: int,
    ) -> int:
        """Phase-based group scheduling with dynamic phases (sequential).

        Each phase generates up to _PHASE_STEP tokens. Sequences that haven't
        finished are promoted to the next phase with a fresh, tighter cache.

        Returns total num_truncated.
        """
        input_queue: deque[tuple[int, list[int], list[int]]] = deque((i, p, p) for i, p in enumerate(prompts))
        promote_queue: deque[_PromotedSequence] = deque()
        trunc = 0
        phase = 0
        max_total = max_prompt_len + self.max_new_tokens
        total_threshold = min(self._PHASE_STEP, max_total)

        pbar.write(
            f"[phase] Starting grouped generation: {len(input_queue)} prompts, "
            f"phase_step={self._PHASE_STEP} max={self.max_new_tokens}"
        )

        # First phase uses input_queue directly (loop handles VRAM-driven re-queue)
        cache_len = max(total_threshold, max_prompt_len) + 1
        while input_queue:
            trunc += self._run_phase(input_queue, results, total_threshold, promote_queue, pbar, cache_len)
        phase += 1

        # Subsequent phases process promoted sequences
        while promote_queue:
            total_threshold = min(self._PHASE_STEP * (phase + 1), max_total)
            next_queue: deque[_PromotedSequence] = deque()
            is_last = total_threshold >= max_total
            trunc += self._run_promoted_phase(
                promote_queue, results, total_threshold, None if is_last else next_queue, pbar, max_prompt_len
            )
            promote_queue = next_queue
            phase += 1

        return trunc

    def _run_promoted_phase(
        self,
        promoted_queue: deque[_PromotedSequence],
        results: list[list[int] | None],
        total_threshold: int,
        promote_queue: deque[_PromotedSequence] | None,
        pbar: tqdm,
        max_prompt_len: int,
    ) -> int:
        """Convert promoted sequences to input tuples and run a phase."""
        if not promoted_queue:
            return 0
        input_queue: deque[tuple[int, list[int], list[int]]] = deque()
        for ps in promoted_queue:
            prefill_ids = ps.prompt_ids + ps.generated_ids
            input_queue.append((ps.index, ps.prompt_ids, prefill_ids))
        promoted_queue.clear()
        cache_len = max(total_threshold, max_prompt_len) + 1
        pbar.write(f"[phase] Starting phase: {len(input_queue)} sequences, cache_len={cache_len}")
        trunc = 0
        while input_queue:
            trunc += self._run_phase(input_queue, results, total_threshold, promote_queue, pbar, cache_len)
        return trunc

    def _run_phase(
        self,
        input_queue: deque[tuple[int, list[int], list[int]]],
        results: list[list[int] | None],
        total_threshold: int,
        promote_queue: deque[_PromotedSequence] | None,
        pbar: tqdm,
        max_cache_len: int,
    ) -> int:
        """Run a complete phase: fill → decode → retire/promote until drained.

        Returns the number of truncated sequences.
        """
        self._cache = None

        max_bs = self._vram_reduced_bs if self._vram_reduced_bs is not None else self.max_batch_size
        effective_bs = min(len(input_queue), max_bs)
        if effective_bs != self._effective_batch_size:
            pbar.write(
                f"[perf] Adjusting batch size: {self._effective_batch_size} → {effective_bs}. torch.compile will take time."
            )
            self._effective_batch_size = effective_bs

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._cache = self._init_cache(max_cache_len, effective_bs)
        self._valid_lens = [0] * effective_bs
        active_slots: list[_Slot | None] = [None] * effective_bs
        num_truncated = 0

        step = 0
        step_time_sum = 0.0

        while input_queue or any(s is not None for s in active_slots):
            for slot_idx in range(effective_bs):
                if active_slots[slot_idx] is not None or not input_queue:
                    continue
                result_idx, prompt_ids, prefill_ids = input_queue.popleft()
                # Skip prefill: promote immediately if already at/past threshold
                if promote_queue is not None and len(prefill_ids) >= total_threshold:
                    promote_queue.append(
                        _PromotedSequence(
                            index=result_idx,
                            prompt_ids=prompt_ids,
                            generated_ids=list(prefill_ids[len(prompt_ids) :]),
                        )
                    )
                    continue
                active_slots[slot_idx] = self._prefill(result_idx, prefill_ids, slot_idx, prompt_ids)

            occupied = [(i, s) for i, s in enumerate(active_slots) if s is not None]
            if not occupied:
                break

            slots_only = [s for _, s in occupied]
            max_active_len = max(self._valid_lens[s.batch_idx] for s in slots_only)

            step_start = time.perf_counter()
            self._batched_decode(slots_only)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_time_sum += time.perf_counter() - step_start

            if step % 200 == 0:
                avg_step = step_time_sum / (step + 1)
                pbar.write(
                    f"[perf] phase step={step} active={len(occupied)} "
                    f"max_active_len={max_active_len} avg_step={avg_step:.4f}s "
                    f"queue={len(input_queue)}"
                )
            step += 1

            # Periodic safety net: abort phase if free VRAM is critically low
            if torch.cuda.is_available() and step % self._vram_check_interval == 0:
                if self._check_vram_pressure():
                    self._vram_reduced_bs = max(effective_bs // 2, 1)
                    pbar.write(
                        f"[vram] Free VRAM critically low at step {step}. Aborting phase. Reducing batch size for next attempt: {effective_bs} -> {self._vram_reduced_bs}."
                    )
                    self._force_requeue_active(active_slots, input_queue)
                    break

            for slot_idx, slot in occupied:
                last_token = slot.generated_ids[-1]
                done = last_token == self.eos_token_id or len(slot.generated_ids) >= self.max_new_tokens
                promote = (
                    not done
                    and promote_queue is not None
                    and (slot.prompt_len + len(slot.generated_ids)) >= total_threshold
                )

                if done:
                    if last_token != self.eos_token_id:
                        num_truncated += 1
                    results[slot.index] = slot.generated_ids
                    active_slots[slot_idx] = None
                    pbar.update(1)
                elif promote and promote_queue is not None:
                    promote_queue.append(
                        _PromotedSequence(
                            index=slot.index,
                            prompt_ids=slot.prompt_ids,
                            generated_ids=list(slot.generated_ids),
                        )
                    )
                    active_slots[slot_idx] = None

        if step > 0:
            pbar.write(f"[perf] Phase done: {step} decode steps, avg_step={step_time_sum / step:.4f}s")

        return num_truncated

    def _prefill(
        self,
        prompt_idx: int,
        prefill_ids: list[int],
        batch_idx: int,
        original_prompt_ids: list[int] | None = None,
    ) -> _Slot:
        """Run the prefill forward pass to build KV cache and sample the first token.

        For promoted sequences, prefill_ids = original_prompt + previously_generated.
        The returned slot's generated_ids includes the previously generated tokens
        plus the newly sampled token, so the full generation history is preserved.

        Args:
            prompt_idx: Index into the results array.
            prefill_ids: Token IDs to prefill (prompt + any previously generated tokens).
            batch_idx: Which slot in the batch cache to use.
            original_prompt_ids: The original prompt tokens (for promoted sequences).
                If None, prefill_ids is used as the prompt.
        """
        prompt_ids = original_prompt_ids if original_prompt_ids is not None else prefill_ids
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

        # Preserve previously generated tokens for promoted sequences
        prior_generated = list(prefill_ids[len(prompt_ids) :])
        prior_generated.append(next_token.item())

        return _Slot(
            index=prompt_idx,
            prompt_ids=prompt_ids,
            prompt_len=len(prompt_ids),
            batch_idx=batch_idx,
            generated_ids=prior_generated,
            seq_position=seq_len + 1,
        )

    def _batched_decode(self, slots: list[_Slot]) -> None:
        """Run a single batched decode step for all active slots.

        Each slot writes KV to its own cache position (no shared write + compact).
        Uses the pre-allocated cache — zero tensor allocations per step.
        Tensor batch dimension is sized to effective_batch_size (set per phase).
        """
        device = self.model.device

        # max_active_len determines attention mask width and cache view size.
        max_active_len = max(self._valid_lens[s.batch_idx] for s in slots)
        self._cache._active_seq_len = max_active_len

        # Set per-row cache positions so each slot writes KV at its own valid_len.
        bs = self._effective_batch_size
        per_row_positions = torch.zeros(bs, dtype=torch.long, device=device)
        for slot in slots:
            per_row_positions[slot.batch_idx] = self._valid_lens[slot.batch_idx]
        self._cache._per_row_cache_positions = per_row_positions

        # Build input_ids [bs, 1]
        input_ids = torch.full((bs, 1), self.pad_token_id, dtype=torch.long, device=device)
        for slot in slots:
            input_ids[slot.batch_idx, 0] = slot.generated_ids[-1]

        # Build attention_mask [bs, max_active_len + 1]
        attn_mask = torch.zeros(bs, max_active_len + 1, dtype=torch.long, device=device)
        for slot in slots:
            valid_len = self._valid_lens[slot.batch_idx]
            attn_mask[slot.batch_idx, :valid_len] = 1  # valid cached positions
            attn_mask[slot.batch_idx, valid_len] = 1  # new token position

        # cache_position: max_active_len for correct causal mask sizing
        cache_position = torch.tensor([max_active_len], device=device)

        # position_ids [bs, 1]
        position_ids = torch.zeros(bs, 1, dtype=torch.long, device=device)
        for slot in slots:
            position_ids[slot.batch_idx, 0] = slot.seq_position

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
        for slot in slots:
            next_token = self._sample_token(outputs.logits[slot.batch_idx : slot.batch_idx + 1, -1, :])
            slot.generated_ids.append(next_token.item())
            slot.seq_position += 1
            self._valid_lens[slot.batch_idx] += 1

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
