import string
from typing import override

from transformers import PreTrainedTokenizer

from core.datasets.qa_dataset import InvalidAnswerError, QADataset, QADatasetConfig


class GSM8KDirectResponseDataset(QADataset[QADatasetConfig]):
    def __init__(self, tokenizer: PreTrainedTokenizer, config: QADatasetConfig):
        super().__init__(tokenizer, config)

        self.option_ids = list(string.ascii_lowercase)

    @override
    def system_prompt(self, row: dict) -> str:
        return "The following are grade school math word problems. Please, return your answer as a single number (without extra/special symbols) and nothing else."

    @override
    def user_prompt(self, row: dict) -> str:
        question = row["question"]
        return question.strip()

    @override
    def assistant_response(self, row: dict) -> str:
        return self._ground_truth_answer(row)

    @override
    def row_id(self, row: dict) -> str:
        return row["question_id"]

    @override
    def verify_assistant_response(self, row: dict, assistant_response: str) -> tuple[str, bool]:
        parsed_answer = assistant_response.strip().lower()

        if not parsed_answer:
            # Empty after stripping special tokens: the model emitted a thinking/special token
            # instead of an answer letter. Treat as a failed measurement, not a crash.
            raise InvalidAnswerError("empty answer (model likely emitted a thinking/special token)")

        try:
            return parsed_answer, self._ground_truth_answer(row) == parsed_answer
        except:
            return parsed_answer, False

    def _ground_truth_answer(self, row: dict) -> str:
        return str(row["answer"]).strip().lower()
