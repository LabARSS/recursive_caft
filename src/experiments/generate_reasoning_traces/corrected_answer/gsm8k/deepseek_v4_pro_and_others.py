from multiprocessing import freeze_support
from pathlib import Path

from core.datasets.gsm8k.gsm8k_corrected_answer_dataset import GSM8KCorrectedAnswerDataset
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
            "../../../../../data/out/distillation/gsm8k_corrected_answer_deepseek_v4_pro.parquet"
        )
    )

    df = distill_on_dataset(
        DistillationConfig(
            out_filename=out_filename,
            model="deepseek/deepseek-v4-pro",
            dataset=GSM8KCorrectedAnswerDataset(
                tokenizer=None,  # type: ignore[reportArgumentType]
                config=QADatasetConfig(
                    path=str(
                        Path(__file__).parent.joinpath(
                            "../../../../../data/out/distillation/gsm8k_distilled_deepseek_v4_flash_extend_w_pro_head8192_clean.parquet"
                        )
                    ),
                    dataset_id="gsm8k_distilled_deepseek_v4_flash_extend_w_pro_head8192_clean",
                ),
            ),
            field_reasoning="corrected_reasoning",
            regenerate_incorrect=True,
        ),
        distillation_result_writer=CorrectedAnswerResultWriter(),
    )

    # if df["distill_ans_correct"].all():
    #     distill_reasoning = df["distill_reasoning"].astype(str)
    #     corrected_reasoning = df["corrected_reasoning"].astype(str)
    #     has_correction = corrected_reasoning.str.len() > 0
    #     ends_with_correction = pd.Series(
    #         [d.endswith("\n" + c) for d, c in zip(distill_reasoning, corrected_reasoning)],
    #         index=df.index,
    #     )
    #     already_concatenated = has_correction & ends_with_correction

    #     if already_concatenated.any():
    #         print(
    #             f"Skipping concatenation: {already_concatenated.sum()} rows already have corrected_reasoning appended to distill_reasoning."
    #         )
    #     else:
    #         df.loc[has_correction, "distill_reasoning"] = (
    #             distill_reasoning[has_correction] + "\n" + corrected_reasoning[has_correction]
    #         )
    #         df.to_parquet(out_filename, index=False)
