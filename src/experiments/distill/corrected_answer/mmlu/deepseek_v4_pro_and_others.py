from multiprocessing import freeze_support
from pathlib import Path

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

    has_correction = df["corrected_reasoning"].astype(str).str.len() > 0
    df.loc[has_correction, "distill_reasoning"] = (
        df.loc[has_correction, "distill_reasoning"].astype(str)
        + "\n"
        + df.loc[has_correction, "corrected_reasoning"].astype(str)
    )

    df.to_parquet(out_filename, index=False)
