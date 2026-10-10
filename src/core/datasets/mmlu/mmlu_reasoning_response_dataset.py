from typing import override

from core.datasets.mmlu.mmlu_single_token_response_dataset import MMLUSingleTokenResponseDataset


class MMLUReasoningResponseDataset(MMLUSingleTokenResponseDataset):
    @override
    def assistant_response(self, row: dict) -> str:
        reasoning_chain = row["distill_reasoning"].strip()
        answer = str(row["distill_answer"]).strip().lower()
        return f"{self.tokenizer.thinking_start_token}{reasoning_chain}{self.tokenizer.thinking_end_token}{answer}"

    @override
    def verify_assistant_response(self, row: dict, assistant_response: str) -> tuple[str, bool]:
        assert isinstance(self.tokenizer.thinking_end_token, str), (
            "Tokenizer must have a defined thinking_end_token to use MMLUReasoningResponseDataset"
        )
        thinking_end_token_position = assistant_response.find(self.tokenizer.thinking_end_token)
        if thinking_end_token_position == -1:
            return super().verify_assistant_response(row, assistant_response)

        raw = assistant_response[thinking_end_token_position + len(self.tokenizer.thinking_end_token) :]
        for tok in self.tokenizer.all_special_tokens:
            raw = raw.replace(tok, "")
        extracted_answer = raw.strip().lower()

        try:
            return super().verify_assistant_response(row, extracted_answer)
        except:
            return extracted_answer, False
