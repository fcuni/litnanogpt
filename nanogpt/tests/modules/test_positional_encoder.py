import pytest
import torch

from nanogpt.modules.positional_encoder import (LearnedPositionalEncoder, VanillaPositionalEncoder)


def _make_vanilla_encoder(seq_len, emb_dim):
    return VanillaPositionalEncoder(seq_len=seq_len, emb_dim=emb_dim)


@pytest.mark.parametrize("seq_len, emb_dim", [(10, 512), (1, 512), (10, 1)])
def test_vanilla_encoder_returns_correct_dimensions(seq_len, emb_dim):
    pos_encoder = _make_vanilla_encoder(seq_len=seq_len, emb_dim=emb_dim)

    # check when idxs.len == seq_len
    idxs = torch.rand((seq_len,))
    out = pos_encoder(idxs)
    assert out.shape == (seq_len, emb_dim), f"Expected shape {(seq_len, emb_dim)} but got {out.shape}"

    # check when idxs.len < seq_len
    idxs = torch.rand((seq_len - 1,))
    out_short = pos_encoder(idxs)
    assert out_short.shape == (seq_len - 1, emb_dim), f"Expected shape {(seq_len - 1, emb_dim)} but got {out_short.shape}"


@pytest.mark.parametrize("seq_len, emb_dim", [(10, 512), (1, 512), (10, 1)])
def test_vanilla_encoder_returns_correct_values(seq_len, emb_dim):
    # for the vanilla encoder, the encoding is pos_i/seq_len
    pos_encoder = _make_vanilla_encoder(seq_len=seq_len, emb_dim=emb_dim)

    idxs = torch.arange(seq_len)
    out = pos_encoder(idxs)
    expected_out = torch.arange(seq_len).view(-1, 1) / seq_len
    assert torch.allclose(out, expected_out), "Expected the output to be the position divided by the sequence length"

    # check when idxs.len < seq_len
    idxs = torch.arange(seq_len - 1)
    out_short = pos_encoder(idxs)
    expected_out_short = torch.arange(seq_len - 1).view(-1, 1) / seq_len
    assert torch.allclose(out_short, expected_out_short), "Expected the output to be the position divided by the sequence length"

    # the output should be the same independent of the len(idxs)
    out_prefix = out[:seq_len - 1]
    assert torch.allclose(out_prefix, out_short), "Expected the output to be the position divided by the sequence length"


@pytest.mark.parametrize("seq_len, emb_dim", [(10, 512), (1, 512), (10, 1)])
def test_learned_positonal_encoder_returns_correct_dimensions(seq_len, emb_dim):
    pos_encoder = LearnedPositionalEncoder(seq_len=seq_len, emb_dim=emb_dim)

    idxs = torch.rand((seq_len,)).long()
    out = pos_encoder(idxs)
    assert out.shape == (seq_len, emb_dim), f"Expected shape {(seq_len, emb_dim)} but got {out.shape}"

    # check when idxs.len < seq_len
    idxs = torch.rand((seq_len - 1,)).long()
    out_short = pos_encoder(idxs)
    assert out_short.shape == (seq_len - 1, emb_dim), f"Expected shape {(seq_len - 1, emb_dim)} but got {out_short.shape}"


@pytest.mark.parametrize("pos_encoder_fn", [VanillaPositionalEncoder, LearnedPositionalEncoder])
def test_positional_encoder_raises_error_when_input_ndim_not_1(pos_encoder_fn):
    dummy_encoder = pos_encoder_fn(seq_len=1, emb_dim=1)

    idxs = torch.rand((1, 1)).long()    # 2D tensor, should only be 1D

    with pytest.raises(AssertionError):
        dummy_encoder(idxs)
