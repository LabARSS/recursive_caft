from multiprocessing import freeze_support
from pathlib import Path

import pandas as pd

from core.datasets.mmlu.mmlu_corrected_answer_dataset import MMLUCorrectedAnswerDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.distillation.distill import DistillationConfig, DistillationResultWriter, distill_on_dataset


class CorrectedAnswerResultWriter(DistillationResultWriter):
    def write_to_df(self, df, config, result):
        df.at[result.index, config.field_ans] = config.dataset.assistant_response(df.iloc[result.index].to_dict())
        df.at[result.index, config.field_reasoning] = result.answer
        df.at[result.index, config.field_ans_correct] = True


if __name__ == "__main__":
    freeze_support()

    out_filename = str(
        Path(__file__).parent.joinpath(
            "../../../../../data/out/distillation/mmlu_corrected_answer_deepseek_v4_pro_and_others.parquet"
        )
    )

    df = distill_on_dataset(
        DistillationConfig(
            out_filename=out_filename,
            model="qwen/qwen3.6-plus",
            dataset=MMLUCorrectedAnswerDataset(
                tokenizer=None,  # type: ignore[reportArgumentType]
                config=QADatasetConfig(
                    path=str(
                        Path(__file__).parent.joinpath(
                            "../../../../../data/out/distillation/mmlu_distilled_deepseek_v4_flash_regenerate_incorrect_w_large.parquet"
                        )
                    ),
                    dataset_id="mmlu_distilled_deepseek_v4_flash_regenerate_incorrect_w_large",
                ),
            ),
            field_reasoning="corrected_reasoning",
            regenerate_incorrect=True,
        ),
        distillation_result_writer=CorrectedAnswerResultWriter(),
    )

    if df["distill_ans_correct"].all():
        distill_reasoning = df["distill_reasoning"].astype(str)
        corrected_reasoning = df["corrected_reasoning"].astype(str)
        has_correction = corrected_reasoning.str.len() > 0
        ends_with_correction = pd.Series(
            [d.endswith("\n" + c) for d, c in zip(distill_reasoning, corrected_reasoning)],
            index=df.index,
        )
        already_concatenated = has_correction & ends_with_correction

        if already_concatenated.any():
            print(
                f"Skipping concatenation: {already_concatenated.sum()} rows already have corrected_reasoning appended to distill_reasoning."
            )
        else:
            df.loc[has_correction, "distill_reasoning"] = (
                distill_reasoning[has_correction] + "\n" + corrected_reasoning[has_correction]
            )
            df.to_parquet(out_filename, index=False)
