"""Teacher-entropy build (root of Fix C): the merged file must carry the distill
columns and a `teacher_entropy` average, and must guard against missing ids."""

import pandas as pd
import pytest

from core.utils.datasets import add_average_column, merge_mmlu_on_question_id

EXTRA_COLUMNS = [
    {"entropy_value": "llama_70b_entropy_value"},
    {"entropy_value": "qwen_72b_entropy_value"},
]


def _aggregate(df):
    return add_average_column(df, "llama_70b_entropy_value", "qwen_72b_entropy_value", "teacher_entropy")


def _write(tmp_path, name, df):
    path = tmp_path / name
    df.to_parquet(path)
    return path


def test_merge_carries_distill_and_builds_teacher_entropy(tmp_path):
    main = pd.DataFrame(
        {
            "question_id": [1, 2, 3],
            "question": ["q1", "q2", "q3"],
            "options": ["['a']", "['b']", "['c']"],
            "answer": ["a", "b", "c"],
            "base_cluster": ["x", "y", "z"],
            "distill_reasoning": ["r1", "r2", "r3"],
            "distill_answer": ["a", "b", "c"],
        }
    )
    # Entropy files are a superset of the training question_ids (id 4 is extra).
    llama = pd.DataFrame({"question_id": [1, 2, 3, 4], "entropy_value": [0.2, 0.4, 0.6, 0.9]})
    qwen = pd.DataFrame({"question_id": [1, 2, 3, 4], "entropy_value": [0.4, 0.6, 0.8, 0.9]})

    out = tmp_path / "merged.parquet"
    merged = merge_mmlu_on_question_id(
        main_path=_write(tmp_path, "main.parquet", main),
        extra_paths=[_write(tmp_path, "llama.parquet", llama), _write(tmp_path, "qwen.parquet", qwen)],
        extra_columns=EXTRA_COLUMNS,
        aggregation_function=_aggregate,
        save_path=out,
    )

    assert len(merged) == 3  # left-merge onto main; the extra id 4 is dropped
    for col in [
        "distill_reasoning",
        "distill_answer",
        "teacher_entropy",
        "llama_70b_entropy_value",
        "qwen_72b_entropy_value",
    ]:
        assert col in merged.columns
    teacher_q1 = merged.loc[merged["question_id"] == 1, "teacher_entropy"].iloc[0]
    assert teacher_q1 == pytest.approx((0.2 + 0.4) / 2)
    assert out.exists()


def test_merge_creates_missing_save_path_parent(tmp_path):
    # The experiment writes teacher_entropy.parquet into OUT_PATH before that dir
    # exists (the merge runs before the trainer creates it), so the util must mkdir.
    main = pd.DataFrame(
        {"question_id": [1, 2], "distill_reasoning": ["r1", "r2"], "distill_answer": ["a", "b"]}
    )
    llama = pd.DataFrame({"question_id": [1, 2], "entropy_value": [0.2, 0.4]})
    qwen = pd.DataFrame({"question_id": [1, 2], "entropy_value": [0.4, 0.6]})

    out = tmp_path / "does" / "not" / "exist" / "teacher_entropy.parquet"
    assert not out.parent.exists()

    merge_mmlu_on_question_id(
        main_path=_write(tmp_path, "main.parquet", main),
        extra_paths=[_write(tmp_path, "llama.parquet", llama), _write(tmp_path, "qwen.parquet", qwen)],
        extra_columns=EXTRA_COLUMNS,
        aggregation_function=_aggregate,
        save_path=out,
    )

    assert out.exists()


def test_missing_question_id_raises(tmp_path):
    main = pd.DataFrame(
        {"question_id": [1, 2, 3], "distill_reasoning": ["r1", "r2", "r3"], "distill_answer": ["a", "b", "c"]}
    )
    llama = pd.DataFrame({"question_id": [1, 2], "entropy_value": [0.2, 0.4]})  # missing id 3
    qwen = pd.DataFrame({"question_id": [1, 2, 3], "entropy_value": [0.4, 0.6, 0.8]})

    with pytest.raises(ValueError):
        merge_mmlu_on_question_id(
            main_path=_write(tmp_path, "main.parquet", main),
            extra_paths=[_write(tmp_path, "llama.parquet", llama), _write(tmp_path, "qwen.parquet", qwen)],
            extra_columns=EXTRA_COLUMNS,
            aggregation_function=None,
        )
