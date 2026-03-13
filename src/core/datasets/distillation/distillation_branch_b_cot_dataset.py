from typing import override

from transformers import PreTrainedTokenizer

from core.datasets.causal_dataset import CausalDataset, CausalDatasetConfig
from core.prompts.mmlu_cot_answer import answer_marker, cot_answer_prompt, cot_sys_prompt


class DistillationBranchBCoTDataset(CausalDataset[CausalDatasetConfig]):
    def __init__(self, tokenizer: PreTrainedTokenizer, config: CausalDatasetConfig):
        super().__init__(tokenizer, config)

    @override
    def system_prompt(self, row: dict) -> str:
        subject = row["input"].get("subject")
        return cot_sys_prompt(subject or None)

    @override
    def user_prompt(self, row: dict) -> str:
        question = row["input"]["question"]
        options_dict = row["input"]["options"]
        options = [value for _, value in sorted(options_dict.items()) if value]
        return cot_answer_prompt(question, options)

    @override
    def assistant_response(self, row: dict) -> str:
        reasoning = str((row.get("output") or {}).get("thinking") or "").strip()
        answer = str(row["input"]["gold"]).strip().lower()
        parts = []
        if reasoning:
            parts.append(reasoning)
        parts.append(f"{answer_marker[0]}{answer}{answer_marker[1]}")
        return "\n\n".join(parts)

    @override
    def row_id(self, row: dict) -> str:
        return str(row["input"]["question_id"])
