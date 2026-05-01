from multiprocessing import freeze_support
from pathlib import Path

from core.datasets.mmlu.mmlu_corrected_answer_dataset import MMLUCorrectedAnswerDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.distillation.distill import DistillationConfig, DistillationResultWriter, distill_on_dataset


class CorrectedAnswerResultWriter(DistillationResultWriter):
    def write_to_df(self, df, config, result):
        df.at[result.index, config.field_ans] = config.dataset.assistant_response(df.iloc[result.index].to_dict())
        df.at[result.index, config.field_reasoning] = f"{df.at[result.index, 'distill_reasoning']}\n{result.answer}"
        df.at[result.index, config.field_ans_correct] = True


if __name__ == "__main__":
    freeze_support()

    distill_on_dataset(
        DistillationConfig(
            out_filename=str(
                Path(__file__).parent.joinpath(
                    "../../../../../data/out/distillation/mmlu_corrected_answer_deepseek_v4_pro_and_others.parquet"
                )
            ),
            model="deepseek/deepseek-v4-pro",
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
            timeout=600.0,
        ),
        distillation_result_writer=CorrectedAnswerResultWriter(),
    )
