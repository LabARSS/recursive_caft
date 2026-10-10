import os
from pathlib import Path

import pandas as pd
import torch
from pydantic import BaseModel
from pydraconf import PydraConfig
from tqdm import tqdm
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.modeling_utils import PreTrainedModel

from core.complexity_estimation.complexity_estimator import BaseComplexityEstimator
from core.datasets.base_dataset_adapter import TokenizedRow
from core.datasets.qa_dataset import InvalidAnswerError
from core.datasets.qa_dataset_adapter import QADatasetAdapter


class ModelGenerateConfig(BaseModel):
    max_new_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    do_sample: bool = False
    num_beams: int = 1


class ComplexityEstimationRunnerConfig(PydraConfig):
    out_path: str
    answer_field_name: str
    answer_correctness_field_name: str
    generate_config: ModelGenerateConfig
    save_every: int = 100


class ComplexityEstimationRunner:
    def __init__(
        self,
        config: ComplexityEstimationRunnerConfig,
        complexity_estimator: BaseComplexityEstimator[BaseModel],
    ):
        self.config = config
        self.complexity_estimator = complexity_estimator

    def estimate(
        self,
        dataset_adapter: QADatasetAdapter,
        model: PreTrainedModel,
    ):
        invalid_answers = 0
        processed_rows = 0
        new_processed_rows = 0

        if os.path.exists(self.tmp_path()):
            # A partially-completed run periodically flushes progress to .tmp (every save_every rows)
            # and only writes out_path at the very end, deleting .tmp right after. So a leftover .tmp
            # means an interrupted run: resume from it, skipping rows already answered. (out_path is
            # never present here in the normal flow, since it and .tmp don't coexist.)
            print(f"Temporary file {self.tmp_path()} found. Resuming complexity estimation from it.")
            ds = dataset_adapter.process_dataset(path_override=str(self.tmp_path()), strict=False)
        else:
            print(f"No temporary file found. Starting from scratch. Writing to output file {self.config.out_path}.")
            if os.path.exists(self.config.out_path):
                os.unlink(self.config.out_path)

            ds = dataset_adapter.process_dataset(strict=False)

            ds = ds.add_column(self.config.answer_field_name, [None] * len(ds))
            ds = ds.add_column(self.config.answer_correctness_field_name, [None] * len(ds))

        for field_name in self.complexity_estimator.schema.keys():
            if field_name not in ds.column_names:
                ds = ds.add_column(field_name, [None] * len(ds))

        df = ds.to_pandas()

        for index, df_row in tqdm(df.iterrows(), total=len(df)):
            processed_rows += 1

            # A resumed .tmp round-trips missing answers back as NaN (not None), so use a
            # missing-aware check — a plain `is not None` would treat NaN (not-yet-measured) rows as
            # done and skip them, so resume would measure nothing.
            if pd.notna(df_row[self.config.answer_field_name]):
                continue

            new_processed_rows += 1

            row = TokenizedRow(**df_row.to_dict())

            input_ids = torch.tensor(row.input_ids).unsqueeze(0).to(model.device)
            attention_mask = torch.tensor(row.attention_mask).unsqueeze(0).to(model.device)

            outputs: GenerateDecoderOnlyOutput = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict_in_generate=True,
                output_scores=True,
                **self.config.generate_config.model_dump(),
                pad_token_id=dataset_adapter.dataset.tokenizer.pad_token_id,
            )

            input_length = len(row.input_ids)
            response_raw = outputs.sequences[0, input_length:]

            response = dataset_adapter.dataset.tokenizer.decode(response_raw, skip_special_tokens=True).strip()

            try:
                if self.complexity_estimator.verify_model_output:
                    parsed_answer, answer_correctness = dataset_adapter.dataset.verify_assistant_response(
                        df_row.to_dict(), response
                    )
                else:
                    parsed_answer, answer_correctness = response, False

                df.at[index, self.config.answer_field_name] = parsed_answer
                df.at[index, self.config.answer_correctness_field_name] = answer_correctness

                for field_name, field_value in self.complexity_estimator.estimate_row(
                    df_row.to_dict(),
                    row,
                    outputs,
                    parsed_answer,
                    answer_correctness,
                    tokenizer=dataset_adapter.dataset.tokenizer,
                ):
                    df.at[index, field_name] = field_value

                if new_processed_rows < 2:
                    print(f"Row {row.row_id}:")
                    print(f"Input: {row.model_dump()}")
                    print(f"Processed: {df.loc[index]}")
            except InvalidAnswerError:
                # Expected: the model didn't emit a parseable answer (e.g. drifted to a thinking
                # token). Leave entropy_value unset (NaN) so it is backfilled; count it quietly.
                invalid_answers += 1
            except Exception as ex:
                print(f"Error processing row {row.row_id}: {ex}")
                invalid_answers += 1

            if processed_rows % self.config.save_every == 0:
                dataset_adapter.save_processed_dataset(df, path=self.tmp_path().as_posix(), tmp=True)
                print(
                    f"Processing dataset {self.config.out_path}... Processed: {processed_rows}/{len(df)}. Invalid answers: {invalid_answers}"
                )

        df = df.drop(columns=["input_ids", "attention_mask", "labels", "row_id"], errors="ignore")
        dataset_adapter.save_processed_dataset(df, path=self.config.out_path, tmp=False)

        if os.path.exists(self.tmp_path()):
            os.unlink(self.tmp_path())

        print(f"Processed dataset {self.config.out_path}. Total entries: {len(ds)}. Invalid answers: {invalid_answers}")

    def tmp_path(self) -> Path:
        return Path(self.config.out_path).with_suffix(".tmp").resolve()
