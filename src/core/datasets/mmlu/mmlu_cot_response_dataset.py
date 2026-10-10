from typing import override

from transformers import PreTrainedTokenizer

from core.datasets.mmlu.mmlu_single_token_response_dataset import MMLUSingleTokenResponseDataset
from core.datasets.qa_dataset import QADatasetConfig


class MMLUCoTResponseDataset(MMLUSingleTokenResponseDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, config: QADatasetConfig):
        super().__init__(tokenizer, config)

        self.answer_marker = ("[[", "]]")

    @override
    def system_prompt(self, row: dict) -> str:
        subject = row["base_cluster"]
        return f"The following are multiple choice questions about {subject}. Explain your thinking process step-by-step. At the end, choose a correct option letter by strictly following this format: {self.answer_marker[0]}correct_option{self.answer_marker[1]}."

    @override
    def assistant_response(self, row: dict) -> str:
        raise NotImplementedError(
            "MMLUCoTResponseDataset does not implement assistant_response since it is not used for training. Use MMLUReasoningResponseDataset for evaluation instead."
        )

    @override
    def verify_assistant_response(self, row: dict, assistant_response: str) -> tuple[str, bool]:
        answer_start_token_position = assistant_response.find(self.answer_marker[0])
        answer_end_token_position = assistant_response.find(self.answer_marker[1])
        if (
            answer_start_token_position == -1
            or answer_end_token_position == -1
            or answer_end_token_position < answer_start_token_position
        ):
            return "", False

        extracted_answer = (
            assistant_response[answer_start_token_position + len(self.answer_marker[0]) : answer_end_token_position]
            .strip()
            .lower()
        )
        if len(extracted_answer) > 1:
            # Llama 3B is stubborn
            extracted_answer = extracted_answer[0]

        correct_answer = str(row["answer"]).strip().lower()
        try:
            return extracted_answer, correct_answer == extracted_answer
        except:
            return extracted_answer, False
