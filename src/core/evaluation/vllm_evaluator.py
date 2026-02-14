import json
from pathlib import Path

import pandas as pd
from pydraconf import PydraConfig
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.utils.logger import logger
from core.utils.seed import set_seed


class VLLMEvaluatorConfig(PydraConfig):
    model_id: str
    checkpoint_dirs: list[str]
    eval_datasets: dict[str, CausalDatasetAdapter]
    out_path: str
    max_tokens: int = 1
    temperature: float = 0.0
    tensor_parallel_size: int = 1
    seed: int = 42


class VLLMEvaluator:
    def __init__(self, config: VLLMEvaluatorConfig):
        self.config = config
        self._llm: LLM | None = None

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            first_ckpt = Path(self.config.checkpoint_dirs[0])
            is_lora = self._is_lora(first_ckpt)

            kwargs = {}
            if is_lora:
                kwargs["enable_lora"] = True
                kwargs["max_lora_rank"] = self._read_lora_rank(first_ckpt)

            self._llm = LLM(
                model=self.config.model_id,
                tensor_parallel_size=self.config.tensor_parallel_size,
                seed=self.config.seed,
                **kwargs,
            )
        return self._llm

    def evaluate(self) -> dict[str, dict]:
        set_seed()

        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        prompts_by_dataset = self._build_all_prompts()

        results: dict[str, dict] = {}

        for ckpt_idx, ckpt_path in enumerate(self.config.checkpoint_dirs):
            ckpt_dir = Path(ckpt_path)
            ckpt_name = ckpt_dir.name
            logger.info("Evaluating checkpoint: {}", ckpt_name)

            lora_request = None
            if self._is_lora(ckpt_dir):
                lora_request = LoRARequest(
                    lora_name=ckpt_name,
                    lora_int_id=ckpt_idx + 1,
                    lora_local_path=ckpt_path,
                )

            dataset_results: dict[str, dict] = {}
            combined_correct = 0
            combined_total = 0

            for ds_name, (messages_list, golds, question_ids) in prompts_by_dataset.items():
                outputs = self.llm.chat(
                    messages=messages_list,
                    sampling_params=sampling_params,
                    lora_request=lora_request,
                )

                metrics = self._compute_metrics(outputs, golds, question_ids)
                dataset_results[ds_name] = metrics
                combined_correct += metrics["correct"]
                combined_total += metrics["total"]

                logger.info("{} — {}: accuracy={:.4f}", ckpt_name, ds_name, metrics["accuracy"])

            combined_accuracy = combined_correct / combined_total if combined_total > 0 else 0.0
            dataset_results["combined"] = {
                "accuracy": combined_accuracy,
                "total": combined_total,
                "correct": combined_correct,
            }

            logger.info("{} — combined: accuracy={:.4f}", ckpt_name, combined_accuracy)

            results[ckpt_name] = dataset_results
            self._save_results(ckpt_name, dataset_results)

        return results

    def _build_all_prompts(self) -> dict[str, tuple[list[list[dict[str, str]]], list[str], list[str]]]:
        prompts_by_dataset: dict[str, tuple[list[list[dict[str, str]]], list[str], list[str]]] = {}

        for ds_name, adapter in self.config.eval_datasets.items():
            df = adapter._load_df()
            messages_list: list[list[dict[str, str]]] = []
            golds: list[str] = []
            question_ids: list[str] = []

            for _, row in df.iterrows():
                messages = [
                    {"role": "system", "content": adapter.system_prompt(row)},
                    {"role": "user", "content": adapter.user_prompt(row)},
                ]
                messages_list.append(messages)
                golds.append(adapter.assistant_response(row))
                question_ids.append(adapter.row_id(row))

            prompts_by_dataset[ds_name] = (messages_list, golds, question_ids)

        return prompts_by_dataset

    def _compute_metrics(
        self,
        outputs: list,
        golds: list[str],
        question_ids: list[str],
    ) -> dict:
        correct = 0
        total = len(outputs)
        incorrect: list[dict] = []

        for output, gold, qid in zip(outputs, golds, question_ids):
            predicted = output.outputs[0].text.strip().lower()
            gold_normalized = gold.strip().lower()

            if predicted == gold_normalized:
                correct += 1
            else:
                incorrect.append(
                    {
                        "question_id": qid,
                        "gold": gold,
                        "predicted": predicted,
                    }
                )

        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
        }

    def _save_results(self, ckpt_name: str, results_by_dataset: dict[str, dict]) -> None:
        out_dir = Path(self.config.out_path) / ckpt_name
        out_dir.mkdir(parents=True, exist_ok=True)

        metrics_summary = {}
        all_incorrect: list[dict] = []

        for ds_name, result in results_by_dataset.items():
            metrics_summary[ds_name] = {k: v for k, v in result.items() if k != "incorrect"}
            for item in result.get("incorrect", []):
                item_with_ds = {**item, "dataset": ds_name}
                all_incorrect.append(item_with_ds)

        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics_summary, f, indent=2)

        if all_incorrect:
            pd.DataFrame(all_incorrect).to_csv(
                out_dir / "incorrect_answers.tsv",
                sep="\t",
                index=False,
            )

        logger.info("Results saved to {}", out_dir)

    @staticmethod
    def _is_lora(checkpoint_dir: Path) -> bool:
        return (checkpoint_dir / "adapter_config.json").exists()

    @staticmethod
    def _read_lora_rank(checkpoint_dir: Path) -> int:
        with open(checkpoint_dir / "adapter_config.json") as f:
            config = json.load(f)
        return config["r"]
