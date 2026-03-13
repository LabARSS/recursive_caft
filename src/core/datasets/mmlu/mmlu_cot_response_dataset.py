import ast
import re
from typing import override

from transformers import PreTrainedTokenizer

from core.datasets.qa_dataset import QADataset, QADatasetConfig
from core.prompts.mmlu_cot_answer import answer_marker, cot_answer_prompt, cot_sys_prompt

FINAL_ANSWER_RE = re.compile(r"\[\[\s*([a-jA-J])\s*\]\]")


class MMLUCoTResponseDataset(QADataset[QADatasetConfig]):
    def __init__(self, tokenizer: PreTrainedTokenizer, config: QADatasetConfig):
        super().__init__(tokenizer, config)

    @override
    def system_prompt(self, row: dict) -> str:
        return cot_sys_prompt(row["base_cluster"])

    @override
    def user_prompt(self, row: dict) -> str:
        options = row["options"]
        if isinstance(options, str):
            options = ast.literal_eval(options)
        return cot_answer_prompt(row["question"], options)

    @override
    def assistant_response(self, row: dict) -> str:
        reasoning = str(row.get("reasoning") or "").strip()
        answer = str(row["answer"]).strip().lower()
        parts = []
        if reasoning:
            parts.append(reasoning)
        parts.append(f"{answer_marker[0]}{answer}{answer_marker[1]}")
        return "\n\n".join(parts)

    @override
    def row_id(self, row: dict) -> str:
        return str(row["question_id"])

    @override
    def verify_assistant_response(self, row: dict, assistant_response: str) -> tuple[str, bool]:
        matches = FINAL_ANSWER_RE.findall(assistant_response)
        parsed_answer = matches[-1].lower() if matches else ""
        return parsed_answer, parsed_answer == str(row["answer"]).strip().lower()
