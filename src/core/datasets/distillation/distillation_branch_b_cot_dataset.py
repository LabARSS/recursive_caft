from typing import override

from transformers import PreTrainedTokenizer

from core.datasets.causal_dataset import CausalDataset, CausalDatasetConfig
from core.prompts.mmlu_single_token_answer import single_token_sys_prompt_with_answer_first_thinking
from core.prompts.thinking_markers import THINKING_START, THINKING_END


class DistillationBranchBCoTDataset(CausalDataset[CausalDatasetConfig]):
    def __init__(self, tokenizer: PreTrainedTokenizer, config: CausalDatasetConfig):
        super().__init__(tokenizer, config)

    @override
    def system_prompt(self, row: dict) -> str:
        inp = row["input"]
        subject = inp.get("subject", "")
        return single_token_sys_prompt_with_answer_first_thinking(subject or None)

    @override
    def user_prompt(self, row: dict) -> str:
        inp = row["input"]
        question = inp["question"]
        options_dict = inp["options"]

        opts = "\n".join([f"{k}. {v}".strip() for k, v in sorted(options_dict.items())])
        return (
            f"Question: {question.strip()}\n\n"
            f"Options:\n{opts}\n\n"
            f"Answer with the option letter first, then provide reasoning inside {THINKING_START}...{THINKING_END} tags."
        )

    @override
    def assistant_response(self, row: dict) -> str:
        gold = str(row["input"]["gold"]).strip().lower()
        out = row["output"]
        thinking = out.get("thinking") or ""

        if thinking:
            return f"{gold}\n{THINKING_START}\n{thinking}\n{THINKING_END}"
        return gold

    @override
    def row_id(self, row: dict) -> str:
        return str(row["input"]["question_id"])
