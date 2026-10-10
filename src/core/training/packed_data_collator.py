"""Collator for pre-packed sequences (see `core.datasets.sequence_packing`).

The blocks are already flattened and padded to a constant `budget` length, so this
just stacks them into tensors. Crucially it emits **no `attention_mask`**: that
omission is what makes transformers' FlashAttention-2 path derive block-diagonal
`cu_seqlens` from `position_ids` (the "padding-free" varlen path), giving
per-document isolation within each pack.
"""

import torch

from core.datasets.sequence_packing import IGNORE_LABEL


class PackedSequenceCollator:
    def __init__(self, label_pad_token_id: int = IGNORE_LABEL):
        # Blocks are already padded to `budget` (pad labels are IGNORE_LABEL), so no
        # cross-example padding happens here; this is kept for parity / clarity.
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self._stack(features, "input_ids"),
            "labels": self._stack(features, "labels"),
            "position_ids": self._stack(features, "position_ids"),
        }

    @staticmethod
    def _stack(features: list[dict], key: str) -> torch.Tensor:
        return torch.stack(
            [
                f[key] if isinstance(f[key], torch.Tensor) else torch.tensor(f[key], dtype=torch.long)
                for f in features
            ]
        )
