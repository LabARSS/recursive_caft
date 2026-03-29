from typing import override

import torch
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.tokenization_utils import PreTrainedTokenizer

from core.complexity_estimation.complexity_estimator import BaseComplexityEstimator
from core.complexity_estimation.entropy.logit_entropy import compute_entropy_from_logits
from core.datasets.base_dataset_adapter import TokenizedRow


class MultiTokenEntropyEstimatorSchema(BaseModel):
    entropy_values: list[float]
    first_token_entropy: float
    last_token_entropy: float
    mean_entropy: float
    max_entropy: float
    min_entropy: float
    scored_tokens: list[int]


class MultiTokenEntropyEstimator(BaseComplexityEstimator[MultiTokenEntropyEstimatorSchema]):
    @property
    @override
    def schema(self) -> dict[str, FieldInfo]:
        return MultiTokenEntropyEstimatorSchema.model_fields

    @override
    def estimate_row(
        self,
        dataset_row: dict,
        input: TokenizedRow,
        outputs: GenerateDecoderOnlyOutput,
        parsed_answer: str,
        answer_correctness: bool,
        tokenizer: PreTrainedTokenizer,
    ) -> MultiTokenEntropyEstimatorSchema:
        # Exclude the EOS token score if generation ended with EOS
        assert outputs.scores is not None, "outputs.scores is None — generation must return scores"
        scores = outputs.scores
        generated_token_ids = outputs.sequences[0].tolist()[len(input.input_ids) :]
        if len(generated_token_ids) > 0 and generated_token_ids[len(scores) - 1] == tokenizer.eos_token_id:
            scores = scores[:-1]

        # (token_length, vocab_size)
        all_token_logits = torch.stack([scores[i][0] for i in range(len(scores))], dim=0)
        entropy = compute_entropy_from_logits(all_token_logits)
        entropy_values = [e.item() for e in entropy]
        return MultiTokenEntropyEstimatorSchema(
            entropy_values=entropy_values,
            first_token_entropy=entropy_values[0],
            last_token_entropy=entropy_values[-1],
            mean_entropy=float(torch.mean(entropy)),
            max_entropy=float(torch.max(entropy)),
            min_entropy=float(torch.min(entropy)),
            scored_tokens=generated_token_ids[: len(scores)],
        )
