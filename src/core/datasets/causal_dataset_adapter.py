from core.datasets.base_dataset_adapter import BaseDatasetAdapter, TokenizedRow
from core.datasets.causal_dataset import CausalDataset


class CausalDatasetAdapter(BaseDatasetAdapter[CausalDataset]):
    def process_row(self, row: dict) -> TokenizedRow:
        input_messages = [
            {"role": "system", "content": self.dataset.system_prompt(row)},
            {"role": "user", "content": self.dataset.user_prompt(row)},
        ]

        full = self.dataset.tokenizer.apply_chat_template(
            input_messages + [{"role": "assistant", "content": self.dataset.assistant_response(row)}],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
        )

        prefix = self.dataset.tokenizer.apply_chat_template(
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

        row_id = self.dataset.row_id(row)

        return TokenizedRow(
            **row,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            row_id=row_id,
        )
