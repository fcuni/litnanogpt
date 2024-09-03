import torch

from nanogpt.modules.llama_linear import LlamaLinear


def test_llama_linear_can_forward():
    llama_ff = LlamaLinear(in_dimensions=10)
    x = torch.randn(10, 10)
    try:
        y = llama_ff(x)
    except Exception as e:
        assert False, f"Error during forward pass: {e}"
    assert y.shape == (10, 10)
