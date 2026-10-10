"""write_thinking_token_rows: persist the trained <think> embedding rows that PEFT's
save_pretrained omits, so the eval/estimation loader can reapply them. Output rows are
written only when input/output embeddings are untied."""

from types import SimpleNamespace

import torch

from core.training.callbacks.save_thinking_token_rows import ROWS_FILENAME, write_thinking_token_rows


class _StubModel:
    def __init__(self, vocab: int, dim: int, tied: bool):
        self._in = torch.nn.Embedding(vocab, dim)
        self._out = torch.nn.Linear(dim, vocab, bias=False)
        self.config = SimpleNamespace(tie_word_embeddings=tied)

    def get_input_embeddings(self):
        return self._in

    def get_output_embeddings(self):
        return self._out


def test_writes_input_and_output_rows_when_untied(tmp_path):
    model = _StubModel(vocab=16, dim=4, tied=False)
    new_ids = [2, 5]

    out = write_thinking_token_rows(model, new_ids, tmp_path)
    assert out == tmp_path / ROWS_FILENAME

    payload = torch.load(out, weights_only=True)
    assert payload["new_ids"] == new_ids
    ids = torch.tensor(new_ids)
    assert torch.equal(payload["input_rows"], model.get_input_embeddings().weight[ids].cpu())
    assert torch.equal(payload["output_rows"], model.get_output_embeddings().weight[ids].cpu())


def test_omits_output_rows_when_tied(tmp_path):
    model = _StubModel(vocab=16, dim=4, tied=True)
    payload = torch.load(write_thinking_token_rows(model, [1], tmp_path), weights_only=True)
    assert "output_rows" not in payload
    assert payload["new_ids"] == [1]
