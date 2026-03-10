from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import DynamicCache, PreTrainedModel, PreTrainedTokenizer


@dataclass
class _Slot:
    index: int
    prompt_len: int
    generated_ids: list[int] = field(default_factory=list)
    cache: DynamicCache = field(default_factory=DynamicCache)
    seq_position: int = 0  # total tokens seen = prompt_len + len(generated_ids)


def _right_pad_cache(cache: DynamicCache, target_len: int) -> DynamicCache:
    """Pad KV cache tensors along seq dim (dim=-2) with zeros to target_len."""
    current_len = cache.get_seq_length()
    if current_len == target_len:
        return cache
    pad_len = target_len - current_len
    padded = DynamicCache()
    for layer_idx in range(len(cache)):
        k = cache.key_cache[layer_idx]  # [1, H, T, D]
        v = cache.value_cache[layer_idx]
        k_pad = F.pad(k, (0, 0, 0, pad_len))  # pad dim=-2
        v_pad = F.pad(v, (0, 0, 0, pad_len))
        padded.update(k_pad, v_pad, layer_idx)
    return padded


def _trim_cache(cache: DynamicCache, valid_len: int) -> DynamicCache:
    """Trim KV cache to only the first valid_len entries along seq dim."""
    trimmed = DynamicCache()
    for layer_idx in range(len(cache)):
        k = cache.key_cache[layer_idx][:, :, :valid_len, :]
        v = cache.value_cache[layer_idx][:, :, :valid_len, :]
        trimmed.update(k, v, layer_idx)
    return trimmed


class ContinuousBatchGenerator:
    """Token-by-token generation with continuous batching via model.forward().

    Maintains a pool of active slots. Empty slots are filled from a queue of
    pending prompts. Each slot holds its own DynamicCache (batch_size=1).

    Prefill runs individually per prompt (no padding waste). Decode batches
    all active slots into a single forward() call by padding KV caches to
    equal length and using an attention mask to ignore padded positions.
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

    @torch.no_grad()
    def generate(self, prompts: list[list[int]]) -> list[list[int]]:
        """Generate responses for a list of prompts using continuous batching.

        Args:
            prompts: List of token ID sequences (one per sample).

        Returns:
            List of generated token ID sequences (excluding prompt), same order
            as input.
        """
        results: list[list[int] | None] = [None] * len(prompts)
        queue: deque[tuple[int, list[int]]] = deque((i, p) for i, p in enumerate(prompts))
        active_slots: list[_Slot | None] = [None] * self.max_batch_size

        while queue or any(s is not None for s in active_slots):
            # FILL: prefill empty slots with new prompts (batch_size=1 each)
            for slot_idx in range(self.max_batch_size):
                if active_slots[slot_idx] is not None or not queue:
                    continue
                prompt_idx, prompt_ids = queue.popleft()
                active_slots[slot_idx] = self._prefill(prompt_idx, prompt_ids)

            # Collect occupied slots
            occupied = [(i, s) for i, s in enumerate(active_slots) if s is not None]
            if not occupied:
                break

            # BATCHED DECODE: single forward() call for all active slots
            slots_only = [s for _, s in occupied]
            self._batched_decode(slots_only)

            # RETIRE: check for completed sequences
            for slot_idx, slot in occupied:
                last_token = slot.generated_ids[-1]
                if last_token == self.eos_token_id or len(slot.generated_ids) >= self.max_new_tokens:
                    results[slot.index] = slot.generated_ids
                    active_slots[slot_idx] = None

        return [r if r is not None else [] for r in results]

    def _prefill(self, prompt_idx: int, prompt_ids: list[int]) -> _Slot:
        """Run the prefill forward pass to build KV cache and sample the first token."""
        device = self.model.device
        input_ids = torch.tensor([prompt_ids], device=device)
        seq_len = len(prompt_ids)
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

        next_token = self._sample_token(outputs.logits[:, -1, :])

        return _Slot(
            index=prompt_idx,
            prompt_len=seq_len,
            generated_ids=[next_token.item()],
            cache=outputs.past_key_values,
            seq_position=seq_len + 1,
        )

    def _batched_decode(self, slots: list[_Slot]) -> None:
        """Run a single batched decode step for all active slots."""
        device = self.model.device
        num_slots = len(slots)

        # Cache lengths before padding (needed for trim after forward)
        slot_cache_lens = [s.cache.get_seq_length() for s in slots]
        max_cache_len = max(slot_cache_lens)

        # Pad each slot's KV cache to max_cache_len, then merge into batched cache
        padded_caches = [_right_pad_cache(s.cache, max_cache_len) for s in slots]
        batched_cache = DynamicCache.from_batch_splits(padded_caches)

        # input_ids: last generated token per slot [num_slots, 1]
        input_ids = torch.tensor([[s.generated_ids[-1]] for s in slots], device=device)

        # attention_mask: [num_slots, max_cache_len + 1] (+1 for the new token)
        attn_mask = torch.zeros(num_slots, max_cache_len + 1, dtype=torch.long, device=device)
        for i, slot in enumerate(slots):
            attn_mask[i, :slot_cache_lens[i]] = 1  # valid cached positions
            attn_mask[i, max_cache_len] = 1  # the new token position (appended at end)

        # cache_position: shared across batch, points to where the new KV is appended
        cache_position = torch.tensor([max_cache_len], device=device)

        # position_ids: each slot's actual position (prompt_len + generated so far)
        position_ids = torch.tensor([[s.seq_position] for s in slots], device=device)

        # Single forward() call
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=batched_cache,
            use_cache=True,
        )

        # Sample next token per slot
        for i, slot in enumerate(slots):
            next_token = self._sample_token(outputs.logits[i : i + 1, -1, :])
            slot.generated_ids.append(next_token.item())
            slot.seq_position += 1

        # Split updated cache back to per-slot, trim padding
        updated_splits = outputs.past_key_values.batch_split(num_slots, split_size=1)
        for i, slot in enumerate(slots):
            valid_len = slot_cache_lens[i] + 1  # original cache len + 1 new token
            slot.cache = _trim_cache(updated_splits[i], valid_len)

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
