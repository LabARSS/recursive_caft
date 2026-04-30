import ast
from multiprocessing import freeze_support
from pathlib import Path

from core.distillation.distill import distill_on_dataset
from core.utils.correctness import check_answer_correct_mmlu

if __name__ == "__main__":
    freeze_support()

    distill_on_dataset(
        in_filename=str(
            Path(__file__).parent.joinpath(
                "../../../../data/out/distillation/mmlu_distilled_deepseek_v4_flash_extend_w_large.parquet"
            )
        ),
        out_filename=str(
            Path(__file__).parent.joinpath(
                "../../../../data/out/distillation/mmlu_distilled_deepseek_v4_flash_regenerate_incorrect_w_large.parquet"
            )
        ),
        get_subject_from_row=lambda row: row["base_cluster"],
        get_question_from_row=lambda row: row["question"],
        get_options_from_row=lambda row: ast.literal_eval(row["options"]),
        check_answer_correct=check_answer_correct_mmlu,
        model="deepseek/deepseek-v4-pro",
        regenerate_incorrect=True,
    )
