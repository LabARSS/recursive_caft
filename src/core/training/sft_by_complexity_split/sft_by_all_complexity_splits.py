from pathlib import Path

from core.training.sft_by_complexity_split.sft_by_single_complexity_split import (
    train_sft_by_complexity_split,
)
from core.utils.training_memory_sandbox import memory_sandbox_worker, run_in_memory_sandbox


TRAIN_POSTFIX = "_train.tsv"
TEST_POSTFIX = "_test.tsv"

# feat: add selective window training and thinking mode to all_complexity_splits

# - Add `only` parameter to train_sft_by_all_complexity_splits that allows
#   selecting specific complexity windows for training instead of running
#   all of them. Accepts flexible input formats: int (0), string ("g2"),
#   or list ([0, 2] / ["g0", "g2"]). When omitted, all windows are
#   trained as before.

# - Add `_normalize_only` helper that converts any supported input format
#   into a validated set of indices with bounds checking.

# - Add `use_thinking` parameter passthrough to enable thinking mode
#   across all selected splits.


@memory_sandbox_worker
def wrapped_train_sft_by_complexity_split(*args, **kwargs):
    return train_sft_by_complexity_split(*args, **kwargs)


def _normalize_only(only, total_n: int) -> set[int]:
    """
    Convert the `only` argument to a set of indices: {0, 2, ...}.
    Supports: None, int, 'g1', '1', ['g0','g2'], [0, 2].
    Validates index bounds.
    """
    if only is None:
        return set(range(total_n))

    if isinstance(only, (int, str)):
        only = [only]

    idxs = set()
    for x in only:
        if isinstance(x, str):
            x = x.strip().lower()
            if x.startswith("g"):
                x = x[1:]
        i = int(x)
        if not (0 <= i < total_n):
            raise IndexError(f"Window index out of range: {i} (total {total_n})")
        idxs.add(i)
    return idxs


def train_sft_by_all_complexity_splits(
    data_folder_path: str,
    out_path: str,
    model_id: str,
    training_kwargs: dict | None = None,
    *,
    use_lora: bool = False,
    lora_kwargs: dict | None = None,
    use_thinking: bool = False,
    only=None,
):
    p = Path(data_folder_path)

    if not p.is_dir():
        raise NotADirectoryError(f"{p} is not a directory")

    children = [c for c in p.iterdir()]
    children.sort()  # alphabetical, case-sensitive

    train_df_paths = [c for c in children if c.name.endswith(TRAIN_POSTFIX)]
    test_df_paths = [c for c in children if c.name.endswith(TEST_POSTFIX)]

    if not train_df_paths:
        raise FileNotFoundError(f"No files with postfix {TRAIN_POSTFIX} found in {data_folder_path}")

    selected = _normalize_only(only, len(train_df_paths))

    print("Available windows (index -> file):")
    for i, pth in enumerate(train_df_paths):
        print(f"  g{i} -> {pth.name}")
    print(f"Selected for training: {sorted(selected)}")

    for i, train_df_path in enumerate(train_df_paths):
        if i not in selected:
            continue

        split_out_path = Path(out_path).joinpath(f"g{i}")
        print("train_sft_by_complexity_split", train_df_path)
        run_in_memory_sandbox(
            wrapped_train_sft_by_complexity_split,
            out_path=split_out_path,
            model_id=model_id,
            train_df_path=train_df_path,
            test_df_paths=test_df_paths,
            training_kwargs=training_kwargs,
            use_lora=use_lora,
            lora_kwargs=lora_kwargs,
            use_thinking=use_thinking,
        )