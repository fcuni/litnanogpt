"""
Dimensionality nomenclature,
* B: batch size
* L: sequence length
* I: input dimension
* H: number of heads
* K: key, value, query triplet == 3
"""

import pytest
import torch

from nanogpt.modules.parallel_multi_head_attention import ParallelMultiHeadAttention

_INPUT_DIM = 10
_HIDDEN_DIM = 5


def _dummy_mh_attention(seq_len, num_heads, is_causal):
    return ParallelMultiHeadAttention(
        input_dim=_INPUT_DIM,
        hidden_dim=_HIDDEN_DIM,
        sequence_length=seq_len,
        num_heads=num_heads,
        is_causal=is_causal,
    )


@pytest.mark.parametrize("bs, seq_len, num_heads, is_causal", [(1, 10, 2, True), (2, 5, 3, False)])
def test_parallel_multi_head_attention_can_forward_pass(bs, seq_len, num_heads, is_causal):
    mh_attention = _dummy_mh_attention(seq_len=seq_len, num_heads=num_heads, is_causal=is_causal)
    input = torch.rand((bs, seq_len, _INPUT_DIM))
    att_mask = torch.ones_like(input)
    mh_out = mh_attention(input, att_mask)
    assert mh_out is not None, "Expected the forward pass to return a value"


@pytest.mark.parametrize("bs, seq_len, num_heads, is_causal", [(1, 10, 2, True), (2, 5, 3, False)])
def test_parallel_multi_head_attention_returns_correct_dimensions(bs, seq_len, num_heads, is_causal):
    mh_attention = _dummy_mh_attention(seq_len=seq_len, num_heads=num_heads, is_causal=is_causal)

    # input and output dimensions are [B, L, I]
    input = torch.rand((bs, seq_len, _INPUT_DIM))
    att_mask = torch.ones_like(input)
    mh_out = mh_attention(input, att_mask)
    expected_shape = (bs, seq_len, _INPUT_DIM)

    assert mh_out.shape == expected_shape, f"Expected shape {expected_shape} but got {mh_out.shape}"
