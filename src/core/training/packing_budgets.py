"""Per-model sequence-packing budgets.

The packing budget is the largest sequence (tokens per packed block) that fits on a
single H100 during LoRA training -- FA2 + gradient checkpointing + bf16 autocast +
thinking-token embedding training. It is deliberately *larger* than the dataset's
longest single chain so a pack can hold one long doc plus several short ones and still
fill the GPU.

Budgets are per-model because the cross-entropy logits `[seq_len, vocab]` dominate
long-context memory and the three v0 bases have very different vocabs (Qwen ~151k,
Llama ~128k, Phi-4-mini ~200k), so the max sequence that fits differs a lot.

Determined empirically by `src/preprocessing/packing_budget_probe.ipynb`.
TODO(andrey): run that notebook and replace the 0 placeholders below with the
recommended budgets it prints.
"""

# Keyed by model nick (the artifacts/base_models_v0/<nick> directory name).
PACKING_BUDGETS: dict[str, int] = {
    "qwen_3b": 0,  # TODO: fill from packing_budget_probe.ipynb
    "llama_3b": 0,  # TODO: fill from packing_budget_probe.ipynb
    "phi4_mini": 0,  # TODO: fill from packing_budget_probe.ipynb
}


def packing_budget(model_nick: str) -> int:
    """Look up the packing budget for a base model, failing loudly if it is unset.

    Raises rather than silently packing at a wrong/zero budget -- so an experiment that
    forgot to fill in the constant cannot start a doomed run."""
    if model_nick not in PACKING_BUDGETS:
        raise KeyError(
            f"No packing budget registered for {model_nick!r}. "
            f"Known models: {sorted(PACKING_BUDGETS)}."
        )
    budget = PACKING_BUDGETS[model_nick]
    if budget <= 0:
        raise ValueError(
            f"Packing budget for {model_nick!r} is not set yet. "
            "Run src/preprocessing/packing_budget_probe.ipynb and fill in "
            "PACKING_BUDGETS in core/training/packing_budgets.py."
        )
    return budget
