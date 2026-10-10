"""Callback that dumps the <think>/</think> rows of embed_tokens (and lm_head if
untied) into each checkpoint dir alongside the adapter. Needed because PEFT's
save_pretrained writes only adapter weights, while these rows live in the base
model and would otherwise be lost.

Loaded back at eval time by MultiCheckpointEvaluator after attaching the adapter.
"""

from pathlib import Path
from typing import override

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.modeling_utils import PreTrainedModel

ROWS_FILENAME = "thinking_token_rows.pt"


def write_thinking_token_rows(model: PreTrainedModel, new_ids: list[int], target_dir: Path) -> Path:
    """Persist the `new_ids` rows of embed_tokens (and lm_head if untied) into target_dir.

    PEFT's save_pretrained writes only adapter weights, while these trained rows live in
    the base model and would otherwise be lost. The payload schema ({new_ids, input_rows,
    [output_rows]}) is what the eval-time loader reapplies after attaching the adapter.
    Single-sourced here so the checkpoint callback and the resampling-trainer model
    snapshot can't drift.
    """
    ids_t = torch.tensor(new_ids, dtype=torch.long)
    in_w: torch.Tensor = model.get_input_embeddings().weight.detach()  # type: ignore[assignment]
    in_rows = in_w[ids_t].cpu().clone()

    payload: dict = {
        "new_ids": list(new_ids),
        "input_rows": in_rows,
    }

    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    if not tied:
        out_layer = model.get_output_embeddings()
        if out_layer is not None:
            out_w: torch.Tensor = out_layer.weight.detach()  # type: ignore[assignment]
            payload["output_rows"] = out_w[ids_t].cpu().clone()

    out_path = target_dir / ROWS_FILENAME
    torch.save(payload, out_path)
    return out_path


class SaveThinkingTokenRowsCallback(TrainerCallback):
    def __init__(self, new_ids: list[int]):
        assert new_ids, "new_ids must be non-empty"
        self.new_ids = new_ids

    @override
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        model: PreTrainedModel = kwargs["model"]
        # Trainer just wrote checkpoint-<global_step>. Match its naming exactly.
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.exists():
            return

        write_thinking_token_rows(model, self.new_ids, checkpoint_dir)
