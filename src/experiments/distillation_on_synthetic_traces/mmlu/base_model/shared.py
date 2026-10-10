from pathlib import Path
from typing import Literal

from transformers import AutoTokenizer

from core.datasets.mmlu.mmlu_cot_response_dataset import MMLUCoTResponseDataset
from core.datasets.mmlu.mmlu_single_token_response_dataset import MMLUSingleTokenResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.evaluation.evaluator import Evaluator, EvaluatorConfig, GenerationConfig
from core.utils.logger import logger

BASE_OUT_PATH = Path(__file__).parent.joinpath(
    "../../../../../artifacts/distillation_on_synthetic_traces/mmlu/base_model"
)


def evaluate_base_model(base_model_id: str, model_name: str, type: Literal["single_token"] | Literal["cot"]) -> None:
    logger.info(f"Evaluating base model {base_model_id} as epoch 0...")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if type == "single_token":
        dataset_cls = MMLUSingleTokenResponseDataset
        max_new_tokens = 10
    else:
        dataset_cls = MMLUCoTResponseDataset
        max_new_tokens = 8192

    out_path = str(BASE_OUT_PATH / f"{model_name}_{type}")

    base_config = EvaluatorConfig(
        model_path=base_model_id,
        eval_dataset=QADatasetAdapter(
            dataset=dataset_cls(
                config=QADatasetConfig(
                    path=Path(__file__)
                    .parent.joinpath("../../../../../data/out/splits/random/mmlu/test.parquet")
                    .as_posix(),
                    dataset_id="mmlu_random_test",
                ),
                tokenizer=tokenizer,
            ),
            add_thinking_start_token=False,
        ),
        out_path=out_path,
        generation=GenerationConfig(max_new_tokens=max_new_tokens, max_batch_size=256),
    )
    base_results = Evaluator(base_config, tokenizer).evaluate()

    for r in base_results:
        logger.info(f"base_model: accuracy={r.accuracy:.4f} ({r.correct}/{r.total})")
        if r.num_truncated > 0:
            pct = r.num_truncated / r.total * 100
            logger.warning(f"base_model: {r.num_truncated}/{r.total} ({pct:.1f}%) sequences reached max_new_tokens")
