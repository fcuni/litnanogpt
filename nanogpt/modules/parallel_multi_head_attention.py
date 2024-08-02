import torch
import torch.nn as nn
import torch.nn.functional as F


def _use_flash_attention() -> bool:
    "Use flash attention if available. Requires an available GPU and torch >= 2.0"
    has_cuda = torch.cuda.is_available()
    has_torch_2 = hasattr(F, "scaled_dot_product_attention")
    return has_cuda and has_torch_2


def _produce_att_mask(mask_size: int) -> torch.Tensor:
    "Produce an attention mask to impose left-to-right causality constraints. Assumes model att size is [B, H, L, L]"
    ones = torch.ones((mask_size, mask_size))
    # make lower triangular matrix
    tril = torch.tril(ones)
    return tril.view((1, 1, *tril.shape))


class ParallelMultiHeadAttention(nn.Module):
    """
    This class takes num of heads as a dimension and parallelizes the  self-attention computation.

    First, we take some input with dims [B, L, I] and shape it as [B, L, H, K, I]. Where B is the batch size, L is the 
    sequence length, I is the input dimension, H is the number of heads and K=3 is the {key,value,query} triplet. Then, 
    we pass evaluate the B batches in parallel for each head.
    """
    def __init__(self, input_dim: int, hidden_dim: int, sequence_length: int, num_heads: int, dropout: float = 0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        # The matrices below now have to deal with an input of dims [B, L, I] -> [B, L, KxHxh], wehre h is the hidden dimension
        self.qvk_linear = nn.Linear(input_dim, 3 * num_heads * hidden_dim)
        self.scale = hidden_dim**-0.5

        # The projection is then from the concated attention output with
        # dims [B, H x h, I], where h is the hidden dimension
        self.project = nn.Linear(hidden_dim * num_heads, input_dim)

        self.use_flash = _use_flash_attention()
        self.dropout = dropout

        self.register_buffer("_att_mask", _produce_att_mask(sequence_length), persistent=False)

    def _broadcast_qvk(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # expected input dimensionality is [B, L, KxHxh]
        assert x.ndim == 3, f"Expected number of input dimensions to be 3, got {x.ndim}"

        bs, seq_len, _ = x.shape
        q, k, v = x.split(self.hidden_dim * self.num_heads, dim=-1)
        # out dims should be [B, H, L, h]
        # we transpose here because we want to concat over H and we only operate over [L, h] is the causal attention
        q = q.view(bs, seq_len, self.num_heads, self.hidden_dim).transpose(1, 2)
        k = k.view(bs, seq_len, self.num_heads, self.hidden_dim).transpose(1, 2)
        v = v.view(bs, seq_len, self.num_heads, self.hidden_dim).transpose(1, 2)
        return q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.qvk_linear(x)
        q, k, v = self._broadcast_qvk(x)

        if self.use_flash:
            att = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0, is_causal=True, scale=self.scale
            )
        else:
            # using inneficient manual implementation
            # do (q [B, H , L, h] x k.T [B, H, h, L]) = qk [B, H, L, L]
            qk = self.scale * (q @ k.transpose(-2, -1))
            qk_masked = qk.masked_fill(self._att_mask[:, :, :qk.size(-2), :qk.size(-1)], float("-inf"))
            att = F.softmax(qk_masked, dim=-1)
            att = qk @ v
        return att


if __name__ == "__main__":
    mh = ParallelMultiHeadAttention(5, 5, 1)
    t = torch.rand((1, 1, 5))
    print(mh(t))
