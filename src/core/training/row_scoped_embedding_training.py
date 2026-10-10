"""
Row-scoped backprop for the embedding table.

Pattern: when training only a handful of new vocabulary rows (e.g. <think>/</think>),
freezing the rest of the embedding tensor with requires_grad=False is not enough --
on tied-weight models, gradients to the lm_head still flow into the shared tensor.
Instead, we keep the tensor as a leaf with requires_grad=True and register a
gradient hook that zeros every row except `new_ids` in-place. This works for both
tied and untied models. For untied models, two hooks are registered.

Used by:
- EmbeddingInitTrainer (full-model frozen except embeddings; v0 init pass)
- LoRATrainer (LoRA adapters + new-token embedding rows, when the
  train_thinking_token_embeddings flag is set)

Mean-init of the new rows is a separate concern and lives in
core.training.thinking_tokens.setup_thinking_tokens(model=...).
"""

from dataclasses import dataclass

import torch
from transformers.modeling_utils import PreTrainedModel

from core.utils.logger import logger


@dataclass
class RowScopedSnapshot:
    """Pre-training snapshot used to assert non-new rows did not drift.

    `non_new_in` and `non_new_out` are CPU clones of all rows EXCEPT new_ids.
    `non_new_out` is None when input/output embeddings are tied (a single
    snapshot covers both).
    """

    new_ids: list[int]
    non_new_in: torch.Tensor
    non_new_out: torch.Tensor | None


def install_row_scoped_grad(model: PreTrainedModel, new_ids: list[int]) -> RowScopedSnapshot:
    """Make `embed_tokens.weight` (and lm_head.weight if untied) trainable but
    only at rows in `new_ids`. Caller must already have set requires_grad on
    every other parameter to whatever is appropriate for their setup -- this
    function does NOT touch unrelated parameter flags.

    Returns a snapshot to be passed to `assert_no_row_drift` after training.
    """
    assert new_ids, "new_ids must be non-empty"

    in_layer = model.get_input_embeddings()
    in_w: torch.Tensor = in_layer.weight  # type: ignore[assignment]
    out_layer = model.get_output_embeddings()

    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    in_ptr = in_w.data_ptr()
    out_ptr = out_layer.weight.data_ptr() if out_layer is not None else None
    if tied and out_layer is not None:
        assert in_ptr == out_ptr, (
            f"config.tie_word_embeddings=True but input/output embedding tensors are not the same "
            f"storage (in_ptr={in_ptr}, out_ptr={out_ptr}). Refusing to install a single hook -- "
            f"the underlying assumption that one hook covers both paths is broken."
        )
    logger.info(
        f"row_scoped_embedding: new_ids={new_ids}; tied={tied}; "
        f"vocab={in_w.shape[0]}; hidden={in_w.shape[1]}"
    )

    in_w.requires_grad = True
    if not tied and out_layer is not None:
        out_layer.weight.requires_grad = True

    vocab = in_w.shape[0]
    new_ids_t = torch.tensor(new_ids, dtype=torch.long)
    keep_col = torch.zeros(vocab, 1, dtype=torch.bool)
    keep_col[new_ids_t] = True

    def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
        # In-place to avoid allocating a second [vocab, hidden] tensor per
        # backward. Hook return value is the (in-place mutated) grad.
        grad.mul_(keep_col.to(dtype=grad.dtype, device=grad.device))
        return grad

    in_w.register_hook(_mask_grad)
    if not tied and out_layer is not None:
        out_layer.weight.register_hook(_mask_grad)

    keep_mask = torch.zeros(vocab, dtype=torch.bool)
    keep_mask[new_ids_t] = True
    non_new_in = in_w.detach()[~keep_mask].cpu().clone()
    non_new_out: torch.Tensor | None = None
    if not tied and out_layer is not None:
        non_new_out = out_layer.weight.detach()[~keep_mask].cpu().clone()

    return RowScopedSnapshot(new_ids=new_ids, non_new_in=non_new_in, non_new_out=non_new_out)


def assert_no_row_drift(model: PreTrainedModel, snapshot: RowScopedSnapshot) -> None:
    """Verify that every row outside `snapshot.new_ids` is bit-for-bit unchanged
    from when `install_row_scoped_grad` was called. Run after training.
    """
    in_w: torch.Tensor = model.get_input_embeddings().weight.detach()  # type: ignore[assignment]
    vocab = in_w.shape[0]
    keep_mask = torch.zeros(vocab, dtype=torch.bool)
    keep_mask[torch.tensor(snapshot.new_ids)] = True
    assert torch.equal(in_w[~keep_mask].cpu(), snapshot.non_new_in), (
        "Pre-existing input-embedding rows drifted during training -- gradient mask failed"
    )
    if snapshot.non_new_out is not None:
        out_layer = model.get_output_embeddings()
        assert out_layer is not None, "Snapshot has untied output rows but model has no output embedding"
        out_w: torch.Tensor = out_layer.weight.detach()  # type: ignore[assignment]
        assert torch.equal(out_w[~keep_mask].cpu(), snapshot.non_new_out), (
            "Pre-existing output-embedding rows drifted during training -- gradient mask failed"
        )
