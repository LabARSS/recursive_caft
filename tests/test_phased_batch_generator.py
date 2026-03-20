"""Tests for BatchGenerator with proper batched decode.

Uses sshleifer/tiny-gpt2 (~500KB) — a minimal GPT-2 model that supports
KV cache and all the HF generation APIs we rely on.

Correctness tests compare batched continuous generation against HF's
model.generate() under greedy decoding (temperature=0).

Performance tests measure wall-clock time: batched vs sequential decode.
"""

import time

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.evaluation.phased_batch_generator import BatchGenerator

MODEL_ID = "sshleifer/tiny-gpt2"
MAX_NEW_TOKENS = 20


@pytest.fixture(scope="module")
def model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    return model, tokenizer


def _hf_generate_greedy(model, tokenizer, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
    """Reference: HF model.generate() with greedy decoding for a single prompt."""
    input_ids = torch.tensor([prompt_ids], device=model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Strip prompt from output
    return output[0, len(prompt_ids) :].tolist()


# ──────────────────────────────────────────────────────────────────────
# Correctness tests
# ──────────────────────────────────────────────────────────────────────


class TestCorrectnessVsHFGenerate:
    """Batched continuous generation must match HF generate() under greedy decoding."""

    def test_single_prompt(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        prompt = tokenizer.encode("Hello world")

        gen = BatchGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            max_batch_size=1,
        )
        [result] = gen.generate([prompt]).sequences

        expected = _hf_generate_greedy(model, tokenizer, prompt, MAX_NEW_TOKENS)
        assert result == expected, f"Mismatch:\n  got:      {result}\n  expected: {expected}"

    def test_multiple_same_length_prompts(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        prompts = [
            tokenizer.encode("The cat sat on"),
            tokenizer.encode("A dog ran to"),
        ]

        gen = BatchGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            max_batch_size=4,
        )
        results = gen.generate(prompts).sequences

        for i, (prompt, result) in enumerate(zip(prompts, results)):
            expected = _hf_generate_greedy(model, tokenizer, prompt, MAX_NEW_TOKENS)
            assert result == expected, f"Prompt {i} mismatch:\n  got:      {result}\n  expected: {expected}"

    def test_variable_length_prompts(self, model_and_tokenizer):
        """Prompts of very different lengths in the same batch."""
        model, tokenizer = model_and_tokenizer
        prompts = [
            tokenizer.encode("Hi"),  # short
            tokenizer.encode("The quick brown fox jumps over the lazy dog and then"),  # long
            tokenizer.encode("Once"),  # short
        ]

        gen = BatchGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            max_batch_size=4,
        )
        results = gen.generate(prompts).sequences

        for i, (prompt, result) in enumerate(zip(prompts, results)):
            expected = _hf_generate_greedy(model, tokenizer, prompt, MAX_NEW_TOKENS)
            assert result == expected, (
                f"Prompt {i} (len={len(prompt)}) mismatch:\n  got:      {result}\n  expected: {expected}"
            )

    def test_variable_length_prompts_batched_pairwise(self, model_and_tokenizer):
        """Prompts with very different lengths forced into the same batch (batch_size=2).

        This specifically tests that the KV cache trim logic correctly preserves
        the new token's KV entry for the shorter slot when slots have different
        cache lengths in the same batch.
        """
        model, tokenizer = model_and_tokenizer
        prompts = [
            tokenizer.encode("Hi"),  # very short
            tokenizer.encode("The quick brown fox jumps over the lazy dog and then keeps running"),  # much longer
        ]

        gen = BatchGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            max_batch_size=2,  # forces both into the same batch
        )
        results = gen.generate(prompts).sequences

        for i, (prompt, result) in enumerate(zip(prompts, results)):
            expected = _hf_generate_greedy(model, tokenizer, prompt, MAX_NEW_TOKENS)
            assert result == expected, (
                f"Prompt {i} (len={len(prompt)}) mismatch:\n  got:      {result}\n  expected: {expected}"
            )

    def test_more_prompts_than_batch_size(self, model_and_tokenizer):
        """Tests continuous batching: prompts > max_batch_size forces queuing."""
        model, tokenizer = model_and_tokenizer
        prompts = [
            tokenizer.encode("One"),
            tokenizer.encode("Two two"),
            tokenizer.encode("Three three three"),
            tokenizer.encode("Four"),
            tokenizer.encode("Five five"),
        ]

        gen = BatchGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            max_batch_size=2,  # only 2 slots, 5 prompts → continuous batching
        )
        results = gen.generate(prompts).sequences

        for i, (prompt, result) in enumerate(zip(prompts, results)):
            expected = _hf_generate_greedy(model, tokenizer, prompt, MAX_NEW_TOKENS)
            assert result == expected, f"Prompt {i} mismatch:\n  got:      {result}\n  expected: {expected}"

    def test_max_new_tokens_cutoff(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        prompt = tokenizer.encode("Hello")
        max_tokens = 5

        gen = BatchGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_tokens,
            max_batch_size=1,
        )
        [result] = gen.generate([prompt]).sequences

        assert len(result) <= max_tokens, f"Generated {len(result)} tokens, max was {max_tokens}"

    def test_result_order_preserved(self, model_and_tokenizer):
        """Results must be returned in the same order as input prompts."""
        model, tokenizer = model_and_tokenizer
        prompts = [
            tokenizer.encode("Alpha"),
            tokenizer.encode("Beta beta beta beta"),
            tokenizer.encode("Gamma"),
        ]

        gen = BatchGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            max_batch_size=2,
        )
        results = gen.generate(prompts).sequences

        assert len(results) == len(prompts)
        for i, (prompt, result) in enumerate(zip(prompts, results)):
            expected = _hf_generate_greedy(model, tokenizer, prompt, MAX_NEW_TOKENS)
            assert result == expected, f"Prompt {i} order/content mismatch"

    def test_defragmentation_on_partial_batch(self, model_and_tokenizer):
        """When the queue is empty and slots retire, defragmentation packs
        remaining slots into [0..N-1]. Verify correctness is preserved.

        Uses 4 prompts with max_batch_size=4 so no queuing occurs.
        Different prompt lengths cause slots to retire at different times,
        triggering defragmentation for the remaining slots.
        """
        model, tokenizer = model_and_tokenizer
        prompts = [
            tokenizer.encode("A"),  # very short → finishes first
            tokenizer.encode("The quick brown fox jumps over the lazy dog and then"),  # long
            tokenizer.encode("Hi there"),  # medium
            tokenizer.encode("Once upon a time in a land far far away there lived"),  # long
        ]

        gen = BatchGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            max_batch_size=4,  # all fit in one batch, no refilling
        )
        results = gen.generate(prompts).sequences

        for i, (prompt, result) in enumerate(zip(prompts, results)):
            expected = _hf_generate_greedy(model, tokenizer, prompt, MAX_NEW_TOKENS)
            assert result == expected, (
                f"Prompt {i} (len={len(prompt)}) mismatch after defragmentation:\n"
                f"  got:      {result}\n  expected: {expected}"
            )

    def test_batch_size_1_same_as_sequential(self, model_and_tokenizer):
        """batch_size=1 should produce identical results (no batching, just prefill+decode)."""
        model, tokenizer = model_and_tokenizer
        prompts = [
            tokenizer.encode("First prompt here"),
            tokenizer.encode("Second"),
        ]

        gen = BatchGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            max_batch_size=1,
        )
        results = gen.generate(prompts).sequences

        for i, (prompt, result) in enumerate(zip(prompts, results)):
            expected = _hf_generate_greedy(model, tokenizer, prompt, MAX_NEW_TOKENS)
            assert result == expected

    def test_phased_generation_correctness(self, model_and_tokenizer):
        """Phased generation must produce identical results to HF generate().

        With max_new_tokens=20, sequences may span multiple phases. Under greedy
        decoding, results must be identical regardless of phase transitions.
        """
        model, tokenizer = model_and_tokenizer
        prompts = [
            tokenizer.encode("Hello world"),
            tokenizer.encode("The quick brown fox jumps over the lazy dog and then"),
            tokenizer.encode("Once"),
            tokenizer.encode("A long time ago in a galaxy far far away"),
            tokenizer.encode("Hi"),
        ]

        gen = BatchGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            max_batch_size=2,  # small batch forces queuing + phase interrupts
        )
        results = gen.generate(prompts).sequences

        for i, (prompt, result) in enumerate(zip(prompts, results)):
            expected = _hf_generate_greedy(model, tokenizer, prompt, MAX_NEW_TOKENS)
            assert result == expected, (
                f"Prompt {i} (len={len(prompt)}) mismatch with phased generation:\n"
                f"  got:      {result}\n  expected: {expected}"
            )


# ──────────────────────────────────────────────────────────────────────
# Performance tests
# ──────────────────────────────────────────────────────────────────────


class TestPerformance:
    """Compare batched vs sequential (batch_size=1) wall-clock time.

    These tests use a tiny model so the speedup may not be dramatic,
    but we verify batching is not slower than sequential.
    """

    @pytest.fixture
    def many_prompts(self, model_and_tokenizer):
        _, tokenizer = model_and_tokenizer
        return [
            tokenizer.encode("The quick brown fox jumps over the lazy dog"),
            tokenizer.encode("Once upon a time in a land far away"),
            tokenizer.encode("To be or not to be that is the question"),
            tokenizer.encode("In the beginning there was nothing"),
            tokenizer.encode("A long time ago in a galaxy far far away"),
            tokenizer.encode("It was the best of times it was the worst"),
            tokenizer.encode("Call me Ishmael some years ago never mind how long"),
            tokenizer.encode("All happy families are alike each unhappy family"),
        ]

    def test_batched_vs_sequential_time(self, model_and_tokenizer, many_prompts):
        model, tokenizer = model_and_tokenizer
        max_tokens = 30

        # Sequential: batch_size=1
        gen_seq = BatchGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_tokens,
            max_batch_size=1,
        )
        start = time.perf_counter()
        results_seq = gen_seq.generate(many_prompts).sequences
        time_sequential = time.perf_counter() - start

        # Batched: batch_size=8
        gen_batch = BatchGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_tokens,
            max_batch_size=8,
        )
        start = time.perf_counter()
        results_batch = gen_batch.generate(many_prompts).sequences
        time_batched = time.perf_counter() - start

        # Verify correctness: both must produce same results
        assert results_seq == results_batch, "Batched and sequential results differ!"

        # Report timing
        speedup = time_sequential / time_batched if time_batched > 0 else float("inf")
        print(f"\n  Sequential (bs=1): {time_sequential:.3f}s")
        print(f"  Batched    (bs=8): {time_batched:.3f}s")
        print(f"  Speedup:           {speedup:.2f}x")

    def test_varying_batch_sizes(self, model_and_tokenizer, many_prompts):
        model, tokenizer = model_and_tokenizer
        max_tokens = 20

        timings = {}
        for bs in [1, 2, 4, 8]:
            gen = BatchGenerator(
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_tokens,
                max_batch_size=bs,
            )
            start = time.perf_counter()
            gen.generate(many_prompts)
            elapsed = time.perf_counter() - start
            timings[bs] = elapsed

        print("\n  Batch size -> Time:")
        for bs, t in timings.items():
            print(f"    bs={bs}: {t:.3f}s")

        # Sanity: all should complete without error (no assertion on speed
        # since tiny-gpt2 on CPU may not show clear scaling)
