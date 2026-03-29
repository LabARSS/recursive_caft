from core.datasets.base_dataset_adapter import BaseDatasetAdapter, TokenizedRow
from core.datasets.qa_dataset import QADataset


class QADatasetAdapter(BaseDatasetAdapter[QADataset]):
    def process_row(self, row: dict) -> TokenizedRow:
        tokenized = self.dataset.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.dataset.system_prompt(row)},
                {"role": "user", "content": self.dataset.user_prompt(row)},
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        row_id = self.dataset.row_id(row)

        return TokenizedRow(
            **row,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=[],
            row_id=row_id,
        )
