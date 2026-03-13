import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch
from tqdm import tqdm
from transformers import DynamicCache, PreTrainedModel, PreTrainedTokenizer
from transformers.cache_utils import _static_cache_update

from core.utils.logger import logger


@dataclass
class GenerationResult:
    sequences: list[list[int]]
    num_truncated: int
    total: int


@dataclass
class _Slot:
    index: int
    prompt_len: int
    batch_idx: int
    generated_ids: list[int] = field(default_factory=list)
    seq_position: int = 0  # total tokens seen = prompt_len + len(generated_ids)


class _PreAllocatedBatchCache(DynamicCache):
    """Pre-allocated KV cache that updates in-place via index_copy_.

    Subclasses DynamicCache so models take the DynamicCache code path in
    _update_causal_mask (target_length = attention_mask.shape[-1]).
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
    ) -> None:
        super().__init__()
        self._seen_tokens = 0
        self._active_seq_len = 0
        self._per_row_cache_positions: Optional[torch.Tensor] = None
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
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if self._per_row_cache_positions is not None:
            # Per-slot decode: each batch row writes KV to its own position.
            k_cache = self.key_cache[layer_idx]
            v_cache = self.value_cache[layer_idx]
            k_cache[self._batch_indices, :, self._per_row_cache_positions, :] = key_states[:, :, 0, :]
            v_cache[self._batch_indices, :, self._per_row_cache_positions, :] = value_states[:, :, 0, :]
        else:
            # Standard path (prefill): use shared cache_position.
            cache_position = cache_kwargs.get("cache_position") if cache_kwargs else None
            _static_cache_update(
                self.key_cache[layer_idx],
                self.value_cache[layer_idx],
                key_states,
                value_states,
                cache_position,
            )
        # Return a view trimmed to _active_seq_len + 1 so attention only sees
        # valid positions, not the full pre-allocated length.
        seq_end = self._active_seq_len + 1
        return (
            self.key_cache[layer_idx][:, :, :seq_end, :],
            self.value_cache[layer_idx][:, :, :seq_end, :],
        )

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self._active_seq_len

    def get_max_cache_shape(self):
        return None

    def reset_slot(self, slot_idx: int) -> None:
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx][slot_idx].zero_()
            self.value_cache[layer_idx][slot_idx].zero_()


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
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.max_batch_size = max_batch_size
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def _init_cache(self, max_seq_len: int) -> _PreAllocatedBatchCache:
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
            max_batch_size=self.max_batch_size,
            max_cache_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

    @torch.no_grad()
    def generate(self, prompts: list[list[int]]) -> GenerationResult:
        """Generate responses for a list of prompts using continuous batching.

        Args:
            prompts: List of token ID sequences (one per sample).

        Returns:
            GenerationResult with generated sequences and truncation stats.
        """
        results: list[list[int] | None] = [None] * len(prompts)
        queue: deque[tuple[int, list[int]]] = deque((i, p) for i, p in enumerate(prompts))
        active_slots: list[_Slot | None] = [None] * self.max_batch_size
        num_truncated = 0
        pbar = tqdm(total=len(prompts), desc="Generating")

        # Pre-allocate the batched KV cache
        max_prompt_len = max(len(p) for p in prompts)
        max_seq_len = max_prompt_len + self.max_new_tokens
        self._cache = self._init_cache(max_seq_len)
        self._valid_lens = [0] * self.max_batch_size

        step = 0
        draining = False
        step_times: list[float] = []

        while queue or any(s is not None for s in active_slots):
            # FILL: prefill empty slots with new prompts (batch_size=1 each)
            for slot_idx in range(self.max_batch_size):
                if active_slots[slot_idx] is not None or not queue:
                    continue
                prompt_idx, prompt_ids = queue.popleft()
                active_slots[slot_idx] = self._prefill(prompt_idx, prompt_ids, slot_idx)

            # Collect occupied slots
            occupied = [(i, s) for i, s in enumerate(active_slots) if s is not None]
            if not occupied:
                break

            if not queue and not draining:
                draining = True
                logger.info(
                    f"[perf] Entering drain phase at step {step}, "
                    f"active_slots={len(occupied)}, "
                    f"completed={sum(1 for r in results if r is not None)}/{len(prompts)}"
                )

            # BATCHED DECODE: single forward() call for all active slots
            slots_only = [s for _, s in occupied]
            max_active_len = max(self._valid_lens[s.batch_idx] for s in slots_only)

            step_start = time.perf_counter()
            self._batched_decode(slots_only)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_elapsed = time.perf_counter() - step_start
            step_times.append(step_elapsed)

            # Log every 50 steps during drain, or every 200 steps otherwise
            log_interval = 50 if draining else 200
            if step % log_interval == 0:
                avg_recent = sum(step_times[-10:]) / len(step_times[-10:])
                logger.info(
                    f"[perf] step={step} active_slots={len(occupied)} "
                    f"max_active_len={max_active_len} "
                    f"step_time={step_elapsed:.4f}s avg_last10={avg_recent:.4f}s "
                    f"queue={len(queue)} draining={draining}"
                )

            step += 1

            # RETIRE: check for completed sequences
            for slot_idx, slot in occupied:
                last_token = slot.generated_ids[-1]
                if last_token == self.eos_token_id or len(slot.generated_ids) >= self.max_new_tokens:
                    if last_token != self.eos_token_id:
                        num_truncated += 1
                    results[slot.index] = slot.generated_ids
                    gen_len = len(slot.generated_ids)
                    logger.info(
                        f"[perf] Prompt {slot.index} completed: "
                        f"gen_len={gen_len} prompt_len={slot.prompt_len} "
                        f"draining={draining} active_slots={len(occupied)} step={step}"
                    )
                    active_slots[slot_idx] = None
                    pbar.update(1)

        logger.info(
            f"[perf] Generation done: {len(step_times)} decode steps, "
            f"avg_step={sum(step_times)/len(step_times):.4f}s"
        )
        pbar.close()
        sequences = [r if r is not None else [] for r in results]
        return GenerationResult(sequences=sequences, num_truncated=num_truncated, total=len(prompts))

    def _prefill(self, prompt_idx: int, prompt_ids: list[int], batch_idx: int) -> _Slot:
        """Run the prefill forward pass to build KV cache and sample the first token.

        Prefills directly into a slice of the pre-allocated cache to avoid
        allocating a temporary DynamicCache.
        """
        device = self.model.device
        input_ids = torch.tensor([prompt_ids], device=device)
        seq_len = len(prompt_ids)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        cache_position = torch.arange(seq_len, device=device)

        # Build a single-slot view into the pre-allocated cache so the model
        # writes KV directly into the right batch row.
        prefill_cache = _PreAllocatedBatchCache.__new__(_PreAllocatedBatchCache)
        DynamicCache.__init__(prefill_cache)
        prefill_cache._seen_tokens = 0
        prefill_cache._active_seq_len = seq_len - 1  # update() will see seq_end = seq_len
        prefill_cache._per_row_cache_positions = None
        prefill_cache.key_cache = [
            self._cache.key_cache[l][batch_idx : batch_idx + 1]
            for l in range(len(self._cache))
        ]
        prefill_cache.value_cache = [
            self._cache.value_cache[l][batch_idx : batch_idx + 1]
            for l in range(len(self._cache))
        ]

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=prefill_cache,
            use_cache=True,
        )
        self._valid_lens[batch_idx] = seq_len

        next_token = self._sample_token(outputs.logits[:, -1, :])

        return _Slot(
            index=prompt_idx,
            prompt_len=seq_len,
            batch_idx=batch_idx,
            generated_ids=[next_token.item()],
            seq_position=seq_len + 1,
        )

    def _batched_decode(self, slots: list[_Slot]) -> None:
        """Run a single batched decode step for all active slots.

        Each slot writes KV to its own cache position (no shared write + compact).
        Uses the pre-allocated cache — zero tensor allocations per step.
        Always uses max_batch_size tensors to avoid torch.compile
        recompilation when the number of active slots changes.
        """
        device = self.model.device

        # max_active_len determines attention mask width and cache view size.
        max_active_len = max(self._valid_lens[s.batch_idx] for s in slots)
        self._cache._active_seq_len = max_active_len

        # Set per-row cache positions so each slot writes KV at its own valid_len.
        per_row_positions = torch.zeros(self.max_batch_size, dtype=torch.long, device=device)
        for slot in slots:
            per_row_positions[slot.batch_idx] = self._valid_lens[slot.batch_idx]
        self._cache._per_row_cache_positions = per_row_positions

        # Build input_ids [max_batch_size, 1]
        input_ids = torch.full(
            (self.max_batch_size, 1), self.pad_token_id, dtype=torch.long, device=device
        )
        for slot in slots:
            input_ids[slot.batch_idx, 0] = slot.generated_ids[-1]

        # Build attention_mask [max_batch_size, max_active_len + 1]
        attn_mask = torch.zeros(
            self.max_batch_size, max_active_len + 1, dtype=torch.long, device=device
        )
        for slot in slots:
            valid_len = self._valid_lens[slot.batch_idx]
            attn_mask[slot.batch_idx, :valid_len] = 1  # valid cached positions
            attn_mask[slot.batch_idx, valid_len] = 1  # new token at slot's own position

        # cache_position: max_active_len for correct causal mask sizing
        cache_position = torch.tensor([max_active_len], device=device)

        # position_ids [max_batch_size, 1]
        position_ids = torch.zeros(self.max_batch_size, 1, dtype=torch.long, device=device)
        for slot in slots:
            position_ids[slot.batch_idx, 0] = slot.seq_position

        # Single forward() call — model writes new KV via per-row cache positions
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=self._cache,
            use_cache=True,
        )

        # Clear per-row positions (revert to standard path for prefill)
        self._cache._per_row_cache_positions = None

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
