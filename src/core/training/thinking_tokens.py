"""
Helper for registering <think>/</think> as special tokens on a tokenizer
and (optionally) resizing + mean-initializing the model embedding table.

Note: tokenizer.thinking_start_token / thinking_end_token are dynamic python
attributes consumed by MMLUReasoningResponseDataset. They are NOT persisted
by tokenizer.save_pretrained — every downstream script that loads a saved
tokenizer must call setup_thinking_tokens(tokenizer) again to re-attach them.
The add_special_tokens call is a no-op when the tokens are already present
(num_added == 0), so it is safe to call repeatedly.
"""

import torch
from transformers import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel

from core.prompts.thinking_markers import THINKING_END, THINKING_START


def setup_thinking_tokens(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel | None = None,
) -> tuple[PreTrainedTokenizer, int]:
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [THINKING_START, THINKING_END]}
    )
    tokenizer.thinking_start_token = THINKING_START
    tokenizer.thinking_end_token = THINKING_END

    if model is not None and num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        _mean_init_new_rows(model, num_added)

    return tokenizer, num_added


def _mean_init_new_rows(model: PreTrainedModel, num_added: int) -> None:
    with torch.no_grad():
        in_emb = model.get_input_embeddings().weight.data
        in_emb[-num_added:] = in_emb[:-num_added].mean(dim=0, keepdim=True)

        if not model.config.tie_word_embeddings:
            out = model.get_output_embeddings()
            if out is not None:
                out_emb = out.weight.data
                out_emb[-num_added:] = out_emb[:-num_added].mean(dim=0, keepdim=True)
