from multiprocessing import freeze_support
from pathlib import Path

from core.datasets.gsm8k.gsm8k_direct_response_dataset import GSM8KDirectResponseDataset, QADatasetConfig
from core.distillation.distill import DistillationConfig, distill_on_dataset

if __name__ == "__main__":
    freeze_support()

    distill_on_dataset(
        DistillationConfig(
            out_filename=str(
                Path(__file__).parent.joinpath(
                    "../../../../../data/out/distillation/gsm8k_distilled_deepseek_v4_flash.parquet"
                )
            ),
            model="deepseek/deepseek-v4-pro",
            dataset=GSM8KDirectResponseDataset(
                tokenizer=None,  # type: ignore[reportArgumentType]
                config=QADatasetConfig(
                    path=str(
                        Path(__file__).parent.joinpath(
                            "../../../../../data/out/distillation/gsm8k_distilled_deepseek_v4_flash_extend_w_large.parquet"
                        )
                    ),
                    dataset_id="gsm8k",
                ),
            ),
        )
    )
