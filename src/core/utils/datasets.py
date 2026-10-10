from pathlib import Path
from typing import Callable

import pandas as pd
from pandas import DataFrame


def merge_mmlu_on_question_id(
    main_path: str | Path,
    extra_paths: list[str | Path],
    extra_columns: list[dict[str, str]],
    aggregation_function: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    save_path: str | Path | None = None,
) -> DataFrame:
    df1 = pd.read_parquet(main_path)
    dfs = [
        pd.read_parquet(path, columns=["question_id", *columns.keys()])
        for path, columns in zip(extra_paths, extra_columns)
    ]

    for i, df2 in enumerate(dfs):
        missing = set(df1["question_id"]) - set(df2["question_id"])
        if missing:
            raise ValueError(
                f"{len(missing)} question_id(s) from {main_path} are missing in {extra_paths[i]}: "
                f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
            )

    dfs = [df2.rename(columns=columns) for df2, columns in zip(dfs, extra_columns)]

    df_merged = df1
    for df2_renamed, df2_extra_columns in zip(dfs, extra_columns):
        df_merged = df_merged.merge(
            df2_renamed[["question_id", *df2_extra_columns.values()]],
            on="question_id",
            how="left",
            validate="one_to_one",
        )

    if aggregation_function:
        df_merged = aggregation_function(df_merged)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df_merged.to_parquet(save_path)

    return df_merged


def add_average_column(
    df: DataFrame,
    col_a: str,
    col_b: str,
    new_col: str,
) -> DataFrame:
    df = df.copy()
    df[new_col] = (df[col_a].astype(float) + df[col_b].astype(float)) / 2.0
    return df


def truncate_column(
    df: DataFrame,
    col: str,
    max_len: int,
) -> DataFrame:
    df = df.copy()
    df[col] = df[col].str.slice(0, max_len)
    return df
