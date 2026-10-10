"""
Helper for registering <think>/</think> as special tokens on a tokenizer
and (optionally) resizing + mean-initializing the model embedding table.

Note: tokenizer.thinking_start_token / thinking_end_token (string forms) and
tokenizer.thinking_start_token_id / thinking_end_token_id (int ids) are dynamic
python attributes consumed by datasets, the evaluator, and training helpers.
They are NOT persisted by tokenizer.save_pretrained — every downstream script
that loads a saved tokenizer must call setup_thinking_tokens(tokenizer) again
to re-attach them. The add_special_tokens call is a no-op when the tokens are
already present (num_added == 0), so it is safe to call repeatedly.
"""

import torch
from transformers import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel

from core.prompts.thinking_markers import THINKING_END, THINKING_START


def setup_thinking_tokens(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel | None = None,
) -> tuple[PreTrainedTokenizer, int]:
    # Important: passing additional_special_tokens to add_special_tokens REPLACES
    # the list on HF tokenizers rather than extending it, which would wipe
    # model-specific specials like <|im_start|>. Extend the current list instead.
    existing = list(tokenizer.additional_special_tokens or [])
    merged = existing + [t for t in (THINKING_START, THINKING_END) if t not in existing]
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": merged})

    tokenizer.thinking_start_token = THINKING_START
    tokenizer.thinking_end_token = THINKING_END
    tokenizer.thinking_start_token_id = tokenizer.convert_tokens_to_ids(THINKING_START)
    tokenizer.thinking_end_token_id = tokenizer.convert_tokens_to_ids(THINKING_END)

    if model is not None:
        # pad_to_multiple_of=64 keeps the lm_head logits stride aligned with
        # what HF's CE / SDPA backward kernels assume. Without it, a tight
        # vocab like Llama-3.2's 128256 + 2 thinking tokens = 128258 trips
        # an inductor compiled-backward stride assert during loss.backward().
        # Padded-vocab models like Phi-4-mini (200064) are unaffected — 64
        # already divides their row count, so resize is a no-op there.
        new_ids = new_token_ids(tokenizer)
        target_len = max(len(tokenizer), max(new_ids) + 1)
        model.resize_token_embeddings(target_len, pad_to_multiple_of=64)
        mean_init_new_rows(model, new_ids)

    return tokenizer, num_added


def new_token_ids(tokenizer: PreTrainedTokenizer) -> list[int]:
    ids: list[int] = []
    for t in (THINKING_START, THINKING_END):
        tid = tokenizer.convert_tokens_to_ids(t)
        assert isinstance(tid, int) and tid >= 0, f"Could not resolve id for {t}: got {tid!r}"
        ids.append(tid)
    return ids


def mean_init_new_rows(model: PreTrainedModel, new_ids: list[int]) -> None:
    """Mean-init the rows at `new_ids` from the mean of all OTHER rows.

    Works for both tight-vocab models (new ids are at the tail of the embedding
    table) and padded-vocab models like Phi-4-mini (new ids sit inside a larger
    pre-existing table; the "existing" rows are everything not in `new_ids`).
    """
    new_ids_t = torch.tensor(new_ids, dtype=torch.long)
    with torch.no_grad():
        in_weight: torch.Tensor = model.get_input_embeddings().weight.data  # type: ignore[assignment]
        _mean_init_tensor(in_weight, new_ids_t)
        if not model.config.tie_word_embeddings:
            out = model.get_output_embeddings()
            if out is not None:
                out_weight: torch.Tensor = out.weight.data  # type: ignore[assignment]
                _mean_init_tensor(out_weight, new_ids_t)


def _mean_init_tensor(weight: torch.Tensor, new_ids: torch.Tensor) -> None:
    vocab = weight.shape[0]
    mask = torch.ones(vocab, dtype=torch.bool, device=weight.device)
    mask[new_ids.to(weight.device)] = False
    mean = weight[mask].mean(dim=0)
    weight[new_ids.to(weight.device)] = mean
