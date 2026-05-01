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
                    "../../../../../data/out/distillation/mmlu_explained_answer_deepseek_v4_flash.parquet"
                )
            ),
            model="deepseek/deepseek-v4-flash",
            dataset=MMLUExplainedAnswerDataset(
                tokenizer=None,  # type: ignore[reportArgumentType]
                config=QADatasetConfig(
                    path=str(Path(__file__).parent.joinpath("../../../../../data/source/mmlu_pro_stem.parquet")),
                    dataset_id="mmlu_pro_stem",
                ),
            ),
            timeout=600.0,
        ),
        distillation_result_writer=ExplainedAnswerResultWriter(),
    )
