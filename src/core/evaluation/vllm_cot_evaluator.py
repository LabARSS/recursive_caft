from typing import override

from core.evaluation.vllm_evaluator import VLLMEvaluator, VLLMEvaluatorConfig
from core.prompts.thinking_markers import THINKING_END


class VLLMCoTEvaluatorConfig(VLLMEvaluatorConfig):
    max_tokens: int = 4096


class VLLMCoTEvaluator(VLLMEvaluator):
    @override
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
            generated_text = output.outputs[0].text
            predicted = self._extract_answer(generated_text)
            gold_normalized = gold.strip().lower()

            if predicted == gold_normalized:
                correct += 1
            else:
                incorrect.append({
                    "question_id": qid,
                    "gold": gold,
                    "predicted": predicted,
                    "full_output": generated_text,
                })

        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
        }

    @staticmethod
    def _extract_answer(text: str) -> str:
        end_idx = text.find(THINKING_END)
        if end_idx == -1:
            return ""
        after_think = text[end_idx + len(THINKING_END):].strip()
        if not after_think:
            return ""
        return after_think[0].lower()
