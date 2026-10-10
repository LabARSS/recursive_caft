import ast
from typing import override

from core.datasets.mmlu.mmlu_single_token_response_dataset import MMLUSingleTokenResponseDataset


class MMLUExplainedAnswerDataset(MMLUSingleTokenResponseDataset):
    @override
    def system_prompt(self, row: dict) -> str:
        subject = row["base_cluster"]
        return f"The following is a multiple choice question about {subject}. The correct answer is provided to you, but your task is to produce a detailed chain-of-thought with step-by-step reasoning that reads as if you were solving the question from scratch — including any natural uncertainty, consideration of multiple options, or self-correction. Do not state or imply that the answer was given to you. End your response with Answer: <letter>."

    @override
    def user_prompt(self, row: dict) -> str:
        question = row["question"]
        options = ast.literal_eval(row["options"])

        options_str = "\n".join(
            [f"{option_id}. {answer}".strip() for option_id, answer in zip(self.option_ids, options)]
        )
        user_prompt = (
            f"Question: {question.strip()}\nOptions:\n{options_str}\nCorrect option:\n{self.assistant_response(row)}\n"
        )
        return user_prompt

    @override
    def assistant_response(self, row: dict) -> str:
        return str(row["answer"]).strip().lower()

    @override
    def row_id(self, row: dict) -> str:
        return str(row["question_id"])

    @override
    def verify_assistant_response(self, row: dict, assistant_response: str) -> tuple[str, bool]:
        return assistant_response, True
