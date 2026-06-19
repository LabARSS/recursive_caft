"""Sequence packing helpers.

A few rows in the distillation traces are 2-3x longer than the median. With
dynamic padding that long tail forces the per-device batch size down so the rare
long batch still fits, which wastes the GPU on the (far more common) short rows.

Packing concatenates the variable-length tokenized rows into fixed `budget`-sized
blocks so every step processes a near-budget sequence of *real* tokens instead of
sizing the whole run for the worst case. The budget is the empirically measured max
sequence that fits on the GPU (see core.training.packing_budgets), which must be
>= the longest single row so it always fits whole in one block.

Per-document isolation (no cross-document attention leakage) is achieved with
FlashAttention-2 varlen: each block carries `position_ids` that restart at 0 for
every member document (and for the trailing pad segment). transformers reads those
resets to build block-diagonal `cu_seqlens` (see `prepare_fa2_from_position_ids`),
which is why the packed batch must be fed with no `attention_mask`.
"""

import bisect

import numpy as np
from datasets import Dataset
from tqdm import tqdm

from core.utils.logger import logger

# Matches the loss-ignore index used by the dataset adapters (-100 on the prompt
# prefix) and torch's CrossEntropyLoss default.
IGNORE_LABEL = -100


def sequence_length_stats(lengths: list[int]) -> dict[str, float]:
    """Summarise the token-length distribution of a tokenized dataset and log it."""
    arr = np.asarray(lengths, dtype=np.int64)
    stats = {
        "count": int(arr.size),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "total_tokens": int(arr.sum()),
    }
    logger.info(
        f"Sequence length stats: count={stats['count']} max={stats['max']} "
        f"mean={stats['mean']:.1f} p50={stats['p50']:.0f} p90={stats['p90']:.0f} "
        f"p99={stats['p99']:.0f} total_tokens={stats['total_tokens']}"
    )
    return stats


def pack_dataset(ds: Dataset, budget: int, pad_token_id: int) -> Dataset:
    """Greedily bin-pack tokenized rows into fixed `budget`-sized blocks.

    Uses best-fit-decreasing (place the largest remaining row into the tightest
    block that still fits) which yields near-full blocks with few bins. The result
    has exactly three columns -- `input_ids`, `labels`, `position_ids` -- each row a
    block of length exactly `budget`:

    - `input_ids`  : member rows concatenated, then padded with `pad_token_id`.
    - `labels`     : member labels concatenated (prompt prefixes already -100), pad
                     region set to -100. The first label of every member is already
                     -100 (chat-template prefix), so the cross-document boundary the
                     loss shift lands on is ignored -- no special handling needed.
    - `position_ids`: range(len) restarted per member document and again for the pad
                      segment, so FA2 treats each as an isolated sequence.

    Deterministic (sorts by length, no RNG) so runs stay reproducible under the
    seeded trainer. `attention_mask`/`row_id` and the original columns are dropped --
    a block has no single-row identity, and FA2 derives masking from `position_ids`.
    """
    input_ids_col = ds["input_ids"]
    labels_col = ds["labels"]
    lengths = [len(x) for x in input_ids_col]

    too_long = [i for i, length in enumerate(lengths) if length > budget]
    if too_long:
        worst = max(too_long, key=lambda i: lengths[i])
        raise ValueError(
            f"{len(too_long)} sequence(s) exceed the packing budget {budget} "
            f"(longest is row {worst} at {lengths[worst]} tokens). "
            "Increase the budget (it must be >= the max sequence length) or check truncation."
        )

    # Best-fit-decreasing. `rem_caps` is kept sorted ascending and `rem_bin` maps each
    # remaining-capacity entry to its bin; bisect finds the smallest bin that still fits.
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    bins: list[list[int]] = []
    rem_caps: list[int] = []
    rem_bin: list[int] = []
    for i in tqdm(order, desc="packing"):
        length = lengths[i]
        k = bisect.bisect_left(rem_caps, length)
        if k < len(rem_caps):
            b = rem_bin.pop(k)
            new_rem = rem_caps.pop(k) - length
            bins[b].append(i)
        else:
            b = len(bins)
            bins.append([i])
            new_rem = budget - length
        kk = bisect.bisect_left(rem_caps, new_rem)
        rem_caps.insert(kk, new_rem)
        rem_bin.insert(kk, b)

    packed_input_ids: list[list[int]] = []
    packed_labels: list[list[int]] = []
    packed_position_ids: list[list[int]] = []
    for bin_docs in bins:
        ids: list[int] = []
        labs: list[int] = []
        pos: list[int] = []
        for i in bin_docs:
            doc_ids = list(input_ids_col[i])
            ids.extend(doc_ids)
            labs.extend(labels_col[i])
            pos.extend(range(len(doc_ids)))
        pad_len = budget - len(ids)
        if pad_len:
            ids.extend([pad_token_id] * pad_len)
            labs.extend([IGNORE_LABEL] * pad_len)
            pos.extend(range(pad_len))
        packed_input_ids.append(ids)
        packed_labels.append(labs)
        packed_position_ids.append(pos)

    fill = sum(lengths) / (len(bins) * budget)
    logger.info(
        f"Packed {len(lengths)} rows into {len(bins)} blocks at budget {budget} "
        f"(avg {len(lengths) / len(bins):.2f} docs/block, {fill:.1%} token fill)"
    )

    return Dataset.from_dict(
        {
            "input_ids": packed_input_ids,
            "labels": packed_labels,
            "position_ids": packed_position_ids,
        }
    )
