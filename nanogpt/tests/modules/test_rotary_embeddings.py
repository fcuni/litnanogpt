import torch

from nanogpt.modules.rotary_embeddings import RotaryEmbeddings


def test_rotary_embeddings_can_forward():
    rot_emb = RotaryEmbeddings(10, 10, 10)
    x = torch.randn(1, 1, 10, 10)
    try:
        y, _ = rot_emb(x, x)
    except Exception as e:
        assert False, f"Error during forward pass: {e}"

    assert y.shape == (1, 1, 10, 10)
