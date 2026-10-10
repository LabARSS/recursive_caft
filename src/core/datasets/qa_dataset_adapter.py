from core.datasets.base_dataset_adapter import BaseDatasetAdapter, BaseDatasetSampler, TokenizedRow
from core.datasets.qa_dataset import QADataset


class QADatasetAdapter(BaseDatasetAdapter[QADataset]):
    def __init__(
        self,
        dataset: QADataset,
        dataset_sampler: BaseDatasetSampler | None = None,
        add_thinking_start_token: bool = False,
    ):
        super().__init__(dataset, dataset_sampler)
        self.add_thinking_start_token = add_thinking_start_token

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

        if self.add_thinking_start_token:
            tid = self.dataset.tokenizer.thinking_start_token_id
            assert isinstance(tid, int) and tid >= 0, (
                "add_thinking_start_token=True but tokenizer has no thinking_start_token_id; "
                "call setup_thinking_tokens(tokenizer) in the experiment script."
            )
            input_ids = input_ids + [tid]
            attention_mask = attention_mask + [1]

        row_id = self.dataset.row_id(row)

        return TokenizedRow(
            **row,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=[],
            row_id=row_id,
        )
