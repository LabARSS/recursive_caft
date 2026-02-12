from abc import abstractmethod

import pandas as pd
from transformers import PreTrainedTokenizer

from core.datasets.base_dataset_adapter import BaseDatasetAdapter, TokenizedRow


class CausalDatasetAdapter(BaseDatasetAdapter):
    @abstractmethod
    def system_prompt(self, row: pd.Series) -> str: ...

    @abstractmethod
    def user_prompt(self, row: pd.Series) -> str: ...

    @abstractmethod
    def assistant_response(self, row: pd.Series) -> str: ...

    @abstractmethod
    def row_id(self, row: pd.Series) -> str: ...

    def process_row(self, row: pd.Series, tokenizer: PreTrainedTokenizer) -> TokenizedRow:
        input_messages = [
            {"role": "system", "content": self.system_prompt(row)},
            {"role": "user", "content": self.user_prompt(row)},
        ]

        full = tokenizer.apply_chat_template(
            input_messages + [{"role": "assistant", "content": self.assistant_response(row)}],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
        )

        prefix = tokenizer.apply_chat_template(
            input_messages,
            tokenize=True,
            add_generation_prompt=True,
        )

        prefix_len = len(prefix)

        input_ids = full["input_ids"]
        assert prefix == input_ids[:prefix_len], (
            "Prefix tokens do not match full tokenization — label mask boundary is incorrect"
        )

        attention_mask = full["attention_mask"]
        labels = [-100] * prefix_len + input_ids[prefix_len:]

        row_id = self.row_id(row)

        return TokenizedRow(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            row_id=row_id,
        )
