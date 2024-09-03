import torch
from datasets.tasks import base
from torch import broadcast_shapes, nn


def _get_freqs(dim: int, max_len: int, base_freq: int) -> torch.Tensor:
    """
    Returns the frequencies for the rotary matrix as polar coordinates, placing the angular frequencies on the unit circle.
    """
    exponents = 2 * torch.arange(0, dim // 2, dtype=torch.float32) / dim
    freqs = 1 / (base_freq**exponents)
    arange_ = torch.arange(max_len)
    freqs = torch.outer(arange_, freqs)
    z = torch.polar(torch.ones_like(exponents), freqs)    # Put nums in unit circle
    return z


def _reshape_for_broadcast(z_tensor: torch.Tensor, x: torch.Tensor):
    """
    Expects the dimensions of x to be [B, H, L, e, 2], where: B is the batch_size, H is the number of heads, L is the
    sequence length and (e,2) is the embedding dimension split in odd and even positions.

    See,
    https://github.com/meta-llama/llama3/blob/main/llama/model.py
    where the Llama people extend this to arbitrary `x.ndim` dimensions.
    """
    assert x.ndim == 5, f"Expected the input tokens to have five dimensions [B, H, L, e, 2], got {x.ndim}"
    assert z_tensor.ndim == 2, f"Expected the frequency matrix to have two dimensions, got {z_tensor.ndim}"

    _, _, seq_len, emb_dim, _ = x.shape
    assert z_tensor.shape == (seq_len, emb_dim)

    broadcast_shape = [1, 1, seq_len, emb_dim, 1]    # reshape to same dims as x
    return z_tensor.view(*broadcast_shape)


class RotaryEmbeddings(nn.Module):
    """
    Implementation of rotary embeddings, see https://arxiv.org/pdf/2104.09864 for more details.

    See also the Llama repo for a more performant implementation:
    https://github.com/meta-llama/llama3/blob/main/llama/model.py
    """
    def __init__(
        self,
        emb_dim: int,
        base_freq: int,
        block_len: int,
    ):
        super().__init__()
        self._freqs = _get_freqs(emb_dim, max_len=block_len, base_freq=base_freq)

    def forward(self, query: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Input is expected to have dims [B, H, L, E]
        *other, _ = query.shape

        # Breaks the emb_dim in half and assigns the first the odd positions to cosine component and the even ones
        # the sine component
        zq_ = torch.view_as_complex(query.float().reshape(*other, -1, 2))    # [B, H, L, e, 2]
        zv_ = torch.view_as_complex(values.float().reshape(*other, -1, 2))    # [B, H, L, e, 2]
        # Reshapes the freqency matrix to do scalar multiplication on the sequence and batch dimensions
        freqs_ = _reshape_for_broadcast(self._freqs, zq_)

        # Output dimension is back to [B, H, L, E]
        q_out = torch.view_as_real(zq_ * freqs_).reshape(*other, -1)
        v_out = torch.view_as_real(zv_ * freqs_).reshape(*other, -1)

        # Guarantee same torch.type for input and output
        return q_out.type_as(query), v_out.type_as(values)
