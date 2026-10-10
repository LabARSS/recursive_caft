from typing import override

from core.datasets.gsm8k.gsm8k_direct_response_dataset import GSM8KDirectResponseDataset


class GSM8KCorrectedAnswerDataset(GSM8KDirectResponseDataset):
    @override
    def system_prompt(self, row: dict) -> str:
        return (
            "The following are grade school math word problems. "
            "You will be shown a question, a partial chain-of-thought that was previously produced for it, "
            "the incorrect answer that chain-of-thought arrived at, and the correct answer. "
            "Continue the partial chain-of-thought from where it left off so that it naturally arrives at "
            "the correct answer. Do not restart the reasoning, do not repeat what was already written, and "
            "do not acknowledge that the previous attempt was wrong or that the correct answer was given to "
            "you. Write the continuation as if you were the same reasoner noticing a mistake or new "
            "consideration mid-thought and revising course. In the end, answer with the correct answer. End your response with Answer: <number>."
        )

    @override
    def user_prompt(self, row: dict) -> str:
        question = row["question"]
        original_reasoning = str(row["distill_reasoning"]).strip()
        original_answer = str(row["distill_answer"]).strip().lower()

        user_prompt = (
            f"Question: {question.strip()}\n"
            f"Partial reasoning so far:\n{original_reasoning}\n"
            f"Incorrect option this reasoning led to: {original_answer}\n"
            f"Correct answer: {self.assistant_response(row)}\n"
        )
        return user_prompt
