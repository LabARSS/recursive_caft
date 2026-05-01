import ast
from typing import override

from core.datasets.mmlu.mmlu_explained_answer_dataset import MMLUExplainedAnswerDataset


class MMLUCorrectedAnswerDataset(MMLUExplainedAnswerDataset):
    @override
    def system_prompt(self, row: dict) -> str:
        subject = row["base_cluster"]
        return (
            f"The following are multiple choice questions about {subject}. "
            "You will be shown a question, a partial chain-of-thought that was previously produced for it, "
            "the incorrect option that chain-of-thought arrived at, and the correct option letter. "
            "Continue the partial chain-of-thought from where it left off so that it naturally arrives at "
            "the correct option. Do not restart the reasoning, do not repeat what was already written, and "
            "do not acknowledge that the previous attempt was wrong or that the correct answer was given to "
            "you. Write the continuation as if you were the same reasoner noticing a mistake or new "
            "consideration mid-thought and revising course. In the end, answer with the correct option letter. End your response with Answer: <letter>."
        )

    @override
    def user_prompt(self, row: dict) -> str:
        question = row["question"]
        options = ast.literal_eval(row["options"])
        original_reasoning = str(row["distill_reasoning"]).strip()
        original_answer = str(row["distill_answer"]).strip().lower()

        options_str = "\n".join(
            [f"{option_id}. {answer}".strip() for option_id, answer in zip(self.option_ids, options)]
        )
        user_prompt = (
            f"Question: {question.strip()}\n"
            f"Options:\n{options_str}\n"
            f"Partial reasoning so far:\n{original_reasoning}\n"
            f"Incorrect option this reasoning led to: {original_answer}\n"
            f"Correct option: {self.assistant_response(row)}\n"
        )
        return user_prompt
