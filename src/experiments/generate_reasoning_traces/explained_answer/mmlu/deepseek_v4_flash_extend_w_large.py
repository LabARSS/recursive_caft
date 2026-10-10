from multiprocessing import freeze_support
from pathlib import Path

from core.datasets.mmlu.mmlu_explained_answer_dataset import MMLUExplainedAnswerDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.distillation.distill import DistillationConfig, DistillationResultWriter, distill_on_dataset


class ExplainedAnswerResultWriter(DistillationResultWriter):
    def write_to_df(self, df, config, result):
        df.at[result.index, config.field_ans] = config.dataset.assistant_response(df.iloc[result.index].to_dict())
        df.at[result.index, config.field_reasoning] = result.answer
        df.at[result.index, config.field_ans_correct] = True


if __name__ == "__main__":
    freeze_support()

    distill_on_dataset(
        DistillationConfig(
            out_filename=str(
                Path(__file__).parent.joinpath(
                    "../../../../../data/out/distillation/mmlu_explained_answer_deepseek_v4_flash_extend_w_large.parquet"
                )
            ),
            model="qwen/qwen3.6-plus",
            dataset=MMLUExplainedAnswerDataset(
                tokenizer=None,  # type: ignore[reportArgumentType]
                config=QADatasetConfig(
                    path=str(
                        Path(__file__).parent.joinpath(
                            "../../../../../data/out/distillation/mmlu_explained_answer_deepseek_v4_flash.parquet"
                        )
                    ),
                    dataset_id="mmlu_explained_answer_deepseek_v4_flash",
                ),
            ),
            timeout=600.0,
            dump_every=10,
        ),
        distillation_result_writer=ExplainedAnswerResultWriter(),
    )
