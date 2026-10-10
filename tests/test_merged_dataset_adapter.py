"""MergedDatasetAdapter: shares one source across sub-adapters and reports a summed
sampled size. This is how the entropy_gain experiment mixes the reasoning + single-token
datasets in the resampling trainer."""

from core.dataset_samplers.base_sampler import BaseDatasetSamplerConfig
from core.dataset_samplers.entropy_gain_sampler import EntropyGainSampler
from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.merged_dataset_adapter import MergedDatasetAdapter
from core.datasets.mmlu.mmlu_reasoning_response_dataset import MMLUReasoningResponseDataset
from core.datasets.mmlu.mmlu_single_token_response_dataset import MMLUSingleTokenResponseDataset
from core.datasets.qa_dataset import QADatasetConfig


def _reasoning_adapter(tokenizer, top_k):
    return CausalDatasetAdapter(
        dataset=MMLUReasoningResponseDataset(
            config=QADatasetConfig(path="sentinel", dataset_id="reasoning"), tokenizer=tokenizer
        ),
        dataset_sampler=EntropyGainSampler(BaseDatasetSamplerConfig(top_k=top_k)),
    )


def _single_token_adapter(tokenizer, top_k):
    return CausalDatasetAdapter(
        dataset=MMLUSingleTokenResponseDataset(
            config=QADatasetConfig(path="sentinel", dataset_id="single_token"), tokenizer=tokenizer
        ),
        dataset_sampler=EntropyGainSampler(BaseDatasetSamplerConfig(top_k=top_k)),
    )


def test_sampled_size_sums_subadapters(thinking_tokenizer):
    merged = MergedDatasetAdapter(
        [_reasoning_adapter(thinking_tokenizer, 1024), _single_token_adapter(thinking_tokenizer, 256)]
    )
    assert merged.sampled_size() == 1280


def test_sampled_size_none_when_a_subadapter_has_no_sampler(thinking_tokenizer):
    no_sampler = CausalDatasetAdapter(
        dataset=MMLUReasoningResponseDataset(
            config=QADatasetConfig(path="sentinel", dataset_id="r"), tokenizer=thinking_tokenizer
        )
    )
    merged = MergedDatasetAdapter([_reasoning_adapter(thinking_tokenizer, 8), no_sampler])
    assert merged.sampled_size() is None


def test_process_dataset_shares_override_across_subadapters(thinking_tokenizer, tmp_path, make_resampling_df):
    # Both sub-adapters read the SAME per-epoch parquet, sample their own top_k, and concatenate.
    parquet = tmp_path / "epoch.parquet"
    make_resampling_df(6).to_parquet(parquet)
    merged = MergedDatasetAdapter(
        [_reasoning_adapter(thinking_tokenizer, 4), _single_token_adapter(thinking_tokenizer, 2)]
    )

    ds = merged.process_dataset(path_override=str(parquet))

    assert len(ds) == 6  # 4 reasoning + 2 single-token from the one shared source
    assert "input_ids" in ds.column_names
    assert "labels" in ds.column_names


def test_process_dataset_shuffles_each_subadapter_without_mixing_blocks(
    thinking_tokenizer, tmp_path, make_resampling_df
):
    parquet = tmp_path / "epoch.parquet"
    source = make_resampling_df(8)
    source.to_parquet(parquet)
    merged = MergedDatasetAdapter(
        [_reasoning_adapter(thinking_tokenizer, 8), _single_token_adapter(thinking_tokenizer, 8)]
    )

    ordered = merged.process_dataset(path_override=str(parquet))
    shuffled = merged.process_dataset(path_override=str(parquet), shuffle=True, shuffle_seed=123)
    replay = merged.process_dataset(path_override=str(parquet), shuffle=True, shuffle_seed=123)

    # The entropy sampler may drop non-positive rows, so derive the actual block boundary.
    block_size = len(ordered) // 2
    ordered_reasoning = ordered["row_id"][:block_size]
    ordered_single_token = ordered["row_id"][block_size:]
    shuffled_reasoning = shuffled["row_id"][:block_size]
    shuffled_single_token = shuffled["row_id"][block_size:]

    assert shuffled["row_id"] == replay["row_id"]
    assert shuffled_reasoning != ordered_reasoning
    assert shuffled_single_token != ordered_single_token
    assert set(shuffled_reasoning) == set(ordered_reasoning)
    assert set(shuffled_single_token) == set(ordered_single_token)
    assert shuffled_reasoning != shuffled_single_token
