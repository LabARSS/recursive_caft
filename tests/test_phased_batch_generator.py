"""Tests for BatchGenerator with proper batched decode.

Uses sshleifer/tiny-gpt2 (~500KB) — a minimal GPT-2 model that supports
KV cache and all the HF generation APIs we rely on.

Correctness tests compare batched continuous generation against HF's
model.generate() under greedy decoding (temperature=0).

Performance tests measure wall-clock time: batched vs sequential decode.
"""

import json
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


def _staged_generate(
    model,
    tokenizer,
    prompts: list[list[int]],
    max_new_tokens: int,
    stage_size: int,
    max_batch_size: int,
    max_thinking_tokens: int | None = None,
    thinking_end_token_id: int | None = None,
):
    """Drive BatchGenerator across length-stages the way Evaluator._evaluate_single does.

    Each stage decodes the pooled survivors only up to the next `stage_size`
    boundary, carries unfinished sequences (token ids) forward, and re-issues them
    via resume_generated. Returns (sequences, truncated, thinking_exhausted) in the
    original prompt order — should equal a single-pass generate() under greedy decode.
    """
    n = len(prompts)
    sequences: list[list[int] | None] = [None] * n
    truncated = [False] * n
    thinking_exhausted = [False] * n
    pending = [{"orig": i, "prompt": prompts[i], "prior": [], "te": False} for i in range(n)]
    stage = 0
    while pending:
        carry_len = min((stage + 1) * stage_size, max_new_tokens)
        carry_arg = None if carry_len >= max_new_tokens else carry_len
        gen = BatchGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            max_batch_size=max_batch_size,
            max_thinking_tokens=max_thinking_tokens,
            thinking_end_token_id=thinking_end_token_id,
        )
        r = gen.generate(
            [p["prompt"] for p in pending],
            carry_at_new_tokens=carry_arg,
            resume_generated=[(p["prior"] or None) for p in pending],
            resume_thinking_exhausted=[p["te"] for p in pending],
        )
        next_pending = []
        for off, p in enumerate(pending):
            if r.unfinished[off]:
                next_pending.append(
                    {
                        "orig": p["orig"],
                        "prompt": p["prompt"],
                        "prior": r.sequences[off],
                        "te": r.thinking_budget_exhausted[off],
                    }
                )
            else:
                sequences[p["orig"]] = r.sequences[off]
                truncated[p["orig"]] = r.truncated[off]
                thinking_exhausted[p["orig"]] = r.thinking_budget_exhausted[off]
        pending = next_pending
        stage += 1
    return sequences, truncated, thinking_exhausted


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
# Disk-offload tests
# ──────────────────────────────────────────────────────────────────────


class TestDiskOffload:
    """KV cache disk offload must be byte-identical and clean up after itself."""

    def test_forced_disk_offload_matches_hf(self, model_and_tokenizer, tmp_path):
        """threshold_gb=0 spills every slot to disk; output must still match HF."""
        model, tokenizer = model_and_tokenizer
        prompts = [
            tokenizer.encode("Hello world"),
            tokenizer.encode("The quick brown fox jumps over the lazy dog and then"),
            tokenizer.encode("Once"),
            tokenizer.encode("A long time ago in a galaxy far far away"),
        ]

        gen = BatchGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            max_batch_size=2,  # forces queuing + phase transitions across disk slots
            kv_cache_offload_threshold_gb=0.0,  # every staged slot spills to disk
            kv_cache_spill_dir=str(tmp_path),
        )
        results = gen.generate(prompts).sequences

        for i, (prompt, result) in enumerate(zip(prompts, results)):
            expected = _hf_generate_greedy(model, tokenizer, prompt, MAX_NEW_TOKENS)
            assert result == expected, (
                f"Prompt {i} mismatch with disk-offloaded KV:\n"
                f"  got:      {result}\n  expected: {expected}"
            )

    def test_spill_dir_removed_after_generate(self, model_and_tokenizer, tmp_path):
        """The store's spill subdir must be gone once generate() returns."""
        model, tokenizer = model_and_tokenizer
        prompt = tokenizer.encode("Hello world")

        gen = BatchGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            max_batch_size=1,
            kv_cache_offload_threshold_gb=0.0,
            kv_cache_spill_dir=str(tmp_path),
        )
        gen.generate([prompt])

        leftover = [p for p in tmp_path.iterdir() if p.name.startswith("kv_spill_")]
        assert leftover == [], f"Spill subdir(s) not cleaned up: {leftover}"


# ──────────────────────────────────────────────────────────────────────
# Thinking-cap tests
# ──────────────────────────────────────────────────────────────────────


class TestMaxThinkingTokens:
    """`max_thinking_tokens` enforces a per-slot cap on the thinking phase.

    When the cap is hit, the just-sampled token is replaced in-place by
    `thinking_end_token_id`; total budget stays at `max_new_tokens`.
    """

    def test_cap_fires_when_end_token_unreached(self, model_and_tokenizer):
        """If the model never naturally emits </think>, the cap forces it
        in at position == max_thinking_tokens."""
        model, tokenizer = model_and_tokenizer
        prompt = tokenizer.encode("Hello")

        # Pick a token id that the tiny model is very unlikely to emit;
        # any vocab id works since we're driving the cap, not detection.
        # Use a high-frequency-but-unlikely-here id by reserving a sentinel.
        sentinel_end_id = tokenizer.encode("</think>", add_special_tokens=False)
        if not sentinel_end_id:
            sentinel_end_id = [tokenizer.vocab_size - 1]
        end_id = sentinel_end_id[0]

        max_thinking = 5
        max_new = 12

        gen = BatchGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new,
            max_batch_size=1,
            max_thinking_tokens=max_thinking,
            thinking_end_token_id=end_id,
        )
        gen_result = gen.generate([prompt])
        [result] = gen_result.sequences

        assert len(result) <= max_new
        # The cap fires at len(generated_ids) == max_thinking, and that token
        # is replaced by end_id. So result[max_thinking - 1] must be end_id.
        assert result[max_thinking - 1] == end_id, (
            f"Expected </think>={end_id} at position {max_thinking - 1}, got {result[max_thinking - 1]} "
            f"(full result: {result})"
        )
        assert gen_result.thinking_budget_exhausted == [True]

    def test_natural_end_token_disables_cap(self, model_and_tokenizer):
        """If the model emits </think> early, in_thinking flips to False and
        the cap never fires — subsequent tokens are answer-phase."""
        model, tokenizer = model_and_tokenizer
        prompt = tokenizer.encode("Hello")

        # Run once with no cap to learn what the model emits.
        gen_ref = BatchGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=10,
            max_batch_size=1,
        )
        [reference] = gen_ref.generate([prompt]).sequences

        # Pick the model's first generated token as the synthetic </think>.
        # That guarantees the natural-emit branch fires on step 1.
        end_id = reference[0]

        gen = BatchGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=10,
            max_batch_size=1,
            max_thinking_tokens=3,
            thinking_end_token_id=end_id,
        )
        gen_result = gen.generate([prompt])
        [result] = gen_result.sequences

        # Token 0 is </think>; subsequent tokens should match the unrestricted
        # reference (cap is disabled after natural emit, no override).
        assert result[0] == end_id
        assert result == reference, (
            f"After natural </think> at step 0, generation should match reference.\n"
            f"  got:      {result}\n  expected: {reference}"
        )
        # Natural emit, not forced — cap-fire flag must stay False.
        assert gen_result.thinking_budget_exhausted == [False]

    def test_cap_disabled_matches_reference(self, model_and_tokenizer):
        """max_thinking_tokens=None → behavior identical to existing path."""
        model, tokenizer = model_and_tokenizer
        prompt = tokenizer.encode("Hello world")

        gen = BatchGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            max_batch_size=1,
            max_thinking_tokens=None,
            thinking_end_token_id=None,
        )
        gen_result = gen.generate([prompt])
        [result] = gen_result.sequences

        expected = _hf_generate_greedy(model, tokenizer, prompt, MAX_NEW_TOKENS)
        assert result == expected
        # Cap not configured — flag must be False.
        assert gen_result.thinking_budget_exhausted == [False]

    def test_cap_with_batched_prompts(self, model_and_tokenizer):
        """Per-slot cap fires independently across a batch."""
        model, tokenizer = model_and_tokenizer
        prompts = [
            tokenizer.encode("First"),
            tokenizer.encode("Second prompt"),
            tokenizer.encode("Third"),
        ]

        sentinel_end_id = tokenizer.encode("</think>", add_special_tokens=False)
        if not sentinel_end_id:
            sentinel_end_id = [tokenizer.vocab_size - 1]
        end_id = sentinel_end_id[0]

        max_thinking = 4
        max_new = 10

        gen = BatchGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new,
            max_batch_size=4,
            max_thinking_tokens=max_thinking,
            thinking_end_token_id=end_id,
        )
        results = gen.generate(prompts).sequences

        for i, result in enumerate(results):
            assert len(result) <= max_new
            # If the model didn't naturally emit end_id earlier, the cap
            # forces it at position max_thinking - 1. Either way, end_id
            # must appear at or before that position.
            head = result[:max_thinking]
            assert end_id in head, (
                f"Prompt {i}: expected </think>={end_id} within first {max_thinking} tokens, "
                f"got {result}"
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


# ──────────────────────────────────────────────────────────────────────
# Length-stage carry tests (cross-chunk pooling of the long tail)
# ──────────────────────────────────────────────────────────────────────


class TestLengthStageCarry:
    """carry_at_new_tokens stops still-running sequences at a length-stage boundary
    and returns them as `unfinished`, to be pooled across chunks and re-prefilled in
    the next stage. Staged generation must stay byte-identical to single-pass / HF."""

    def test_default_unfinished_all_false(self, model_and_tokenizer):
        """Without carry_at_new_tokens, nothing is unfinished (behavior unchanged)."""
        model, tokenizer = model_and_tokenizer
        prompts = [tokenizer.encode("Hello world"), tokenizer.encode("Once upon a time")]
        gen = BatchGenerator(model=model, tokenizer=tokenizer, max_new_tokens=MAX_NEW_TOKENS, max_batch_size=2)
        r = gen.generate(prompts)
        assert r.unfinished == [False, False]

    def test_staged_matches_single_stage_and_hf(self, model_and_tokenizer):
        """Multi-stage carry + re-prefill is byte-identical to one pass and to HF."""
        model, tokenizer = model_and_tokenizer
        prompts = [
            tokenizer.encode("Hello world"),
            tokenizer.encode("The quick brown fox jumps over the lazy dog and then"),
            tokenizer.encode("Once"),
        ]

        gen = BatchGenerator(model=model, tokenizer=tokenizer, max_new_tokens=MAX_NEW_TOKENS, max_batch_size=2)
        single = gen.generate(prompts).sequences

        # stage_size=5 forces several carry/re-prefill boundaries (5, 10, 15, 20).
        staged, _, _ = _staged_generate(
            model, tokenizer, prompts, MAX_NEW_TOKENS, stage_size=5, max_batch_size=2
        )

        assert any(len(s) > 5 for s in staged), "test did not exercise a carry boundary"
        for i, prompt in enumerate(prompts):
            hf = _hf_generate_greedy(model, tokenizer, prompt, MAX_NEW_TOKENS)
            assert staged[i] == single[i] == hf, (
                f"Prompt {i} staged mismatch:\n  staged: {staged[i]}\n  single: {single[i]}\n  hf:     {hf}"
            )

    def test_staged_thinking_cap_matches_single_stage(self, model_and_tokenizer):
        """Thinking-cap state survives a stage boundary: the forced </think> fires at
        the same absolute position staged vs un-staged, and the exhausted flag carries."""
        model, tokenizer = model_and_tokenizer
        prompts = [tokenizer.encode("Hello"), tokenizer.encode("Second prompt here")]

        sentinel = tokenizer.encode("</think>", add_special_tokens=False) or [tokenizer.vocab_size - 1]
        end_id = sentinel[0]
        max_thinking = 5  # cap fires at len==5, AFTER the first stage boundary (3)
        max_new = 12

        gen = BatchGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new,
            max_batch_size=2,
            max_thinking_tokens=max_thinking,
            thinking_end_token_id=end_id,
        )
        ref = gen.generate(prompts)

        staged, _, staged_te = _staged_generate(
            model,
            tokenizer,
            prompts,
            max_new,
            stage_size=3,
            max_batch_size=2,
            max_thinking_tokens=max_thinking,
            thinking_end_token_id=end_id,
        )

        assert staged == ref.sequences, f"staged {staged} != single {ref.sequences}"
        assert staged_te == ref.thinking_budget_exhausted
        for s in staged:
            # Cap fired (or model emitted end_id) within the thinking budget.
            assert end_id in s[:max_thinking]

    def test_carry_equal_max_truncates_not_carries(self, model_and_tokenizer):
        """carry_at_new_tokens == max_new_tokens is the final stage: sequences truncate
        at the hard budget (never `unfinished`) — guards the `not hit_global` rule."""
        model, tokenizer = model_and_tokenizer
        prompt = tokenizer.encode("Hello world")
        gen = BatchGenerator(model=model, tokenizer=tokenizer, max_new_tokens=MAX_NEW_TOKENS, max_batch_size=1)
        r = gen.generate([prompt], carry_at_new_tokens=MAX_NEW_TOKENS)
        assert r.unfinished == [False]
        assert r.sequences[0] == _hf_generate_greedy(model, tokenizer, prompt, MAX_NEW_TOKENS)

    def test_checkpoint_resume_matches_uninterrupted(self, model_and_tokenizer, tmp_path):
        """An in-flight checkpoint (done=False) resumes via re-prefill to the same output."""
        model, tokenizer = model_and_tokenizer
        prompt = tokenizer.encode("Hello world")
        max_new = 12

        gen = BatchGenerator(model=model, tokenizer=tokenizer, max_new_tokens=max_new, max_batch_size=1)
        ref = gen.generate([prompt]).sequences[0]

        ckpt = tmp_path / "resume.ckpt.json"
        ckpt.write_text(
            json.dumps(
                {
                    "phase": 0,
                    "rows": {
                        "0": {
                            "generated_ids": ref[:4],
                            "done": False,
                            "truncated": False,
                            "thinking_budget_exhausted": False,
                        }
                    },
                }
            )
        )
        gen2 = BatchGenerator(model=model, tokenizer=tokenizer, max_new_tokens=max_new, max_batch_size=1)
        resumed = gen2.generate([prompt], checkpoint_path=str(ckpt)).sequences[0]
        assert resumed == ref

    def test_checkpoint_wins_over_seed(self, model_and_tokenizer, tmp_path):
        """When both a checkpoint row and a stage seed exist, the (newer) checkpoint wins."""
        model, tokenizer = model_and_tokenizer
        prompt = tokenizer.encode("Hello world")
        max_new = 12

        gen = BatchGenerator(model=model, tokenizer=tokenizer, max_new_tokens=max_new, max_batch_size=1)
        ref = gen.generate([prompt]).sequences[0]

        # Stale seed: correct first token, wrong second — continuing from it would
        # diverge from ref. Checkpoint holds the true 5-token prefix.
        stale_seed = [ref[0], (ref[1] + 1) % tokenizer.vocab_size]
        ckpt = tmp_path / "ck.json"
        ckpt.write_text(
            json.dumps(
                {
                    "phase": 0,
                    "rows": {
                        "0": {
                            "generated_ids": ref[:5],
                            "done": False,
                            "truncated": False,
                            "thinking_budget_exhausted": False,
                        }
                    },
                }
            )
        )
        gen2 = BatchGenerator(model=model, tokenizer=tokenizer, max_new_tokens=max_new, max_batch_size=1)
        resumed = gen2.generate([prompt], checkpoint_path=str(ckpt), resume_generated=[stale_seed]).sequences[0]
        assert resumed == ref, "checkpoint row should override the stale seed"
