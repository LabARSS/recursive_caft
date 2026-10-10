from multiprocessing import freeze_support
from pathlib import Path

from core.datasets.mmlu.mmlu_single_token_response_dataset import MMLUSingleTokenResponseDataset, QADatasetConfig
from core.distillation.distill import DistillationConfig, distill_on_dataset

if __name__ == "__main__":
    freeze_support()

    distill_on_dataset(
        DistillationConfig(
            out_filename=str(
                Path(__file__).parent.joinpath(
                    "../../../../../data/out/distillation/mmlu_distilled_deepseek_v4_flash.parquet"
                )
            ),
            model="deepseek/deepseek-v4-flash",
            dataset=MMLUSingleTokenResponseDataset(
                tokenizer=None,  # type: ignore[reportArgumentType]
                config=QADatasetConfig(
                    path=str(Path(__file__).parent.joinpath("../../../../../data/source/mmlu_pro_stem.parquet")),
                    dataset_id="mmlu_pro_stem",
                ),
            ),
        )
    )
