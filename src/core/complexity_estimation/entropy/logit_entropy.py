import math

import torch


def compute_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized entropy from logits.

    Raw nats are not comparable across models: a student and a proxy teacher with
    different vocabularies (128256 / 131072 / 151936 / 152064 / 200064 here) put their
    mass over different supports, so subtracting one from the other — as the entropy-gain
    sampler does — mixes scales. Dividing by ln|V| (the entropy of the uniform
    distribution over the same support) puts every model on a common [0, 1] scale.

    |V| is taken from the logits themselves, so it always matches the model that produced
    them. Note that a student whose embeddings were resized by `setup_thinking_tokens`
    reports the resized width (e.g. Qwen2.5-3B: 151680 instead of 151936) — a 0.014%
    difference in the divisor versus the offline runs.

    Parameters:
    ----------
    logits : torch.Tensor
        Logits from the model. The vocabulary is the last dimension.

    Returns:
    -------
    torch.Tensor
        Normalized entropy values in [0, 1].
    """
    probabilities = torch.softmax(logits, dim=-1)
    log_probabilities = torch.log(probabilities + 1e-12)
    entropy = -torch.sum(probabilities * log_probabilities, dim=-1)
    return entropy / math.log(logits.shape[-1])
