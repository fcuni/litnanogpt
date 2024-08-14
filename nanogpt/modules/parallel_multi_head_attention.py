from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _use_flash_attention() -> bool:
    """
    Use flash attention if available. Requires torch >= 2.0, if no GPU is available it defaults to torch attention
    implementation
    """
    has_torch_2 = hasattr(F, "scaled_dot_product_attention")
    return has_torch_2


def _produce_att_mask(mask_size: int) -> torch.Tensor:
    "Produce an attention mask to impose left-to-right causality constraints. Assumes model att size is [B, H, L, L]"
    ones = torch.ones((mask_size, mask_size), dtype=torch.bool)
    # make lower triangular matrix
    tril = torch.tril(ones)
    return tril.view((1, 1, *tril.shape))


@dataclass
class AttentionConfig:
    """The defaults are taken from the Nanogpt implementation."""

    input_dim: int = 768
    """Input dimension of the attention layer."""
    hidden_dim: int = 64
    """Hidden dimension of the attention heads."""
    sequence_length: int = 1024
    """Length of the input sequence."""
    num_heads: int = 12
    """Number of attention heads."""
    dropout: float = 0.2
    """Single dropout value used for attention layers, after the softmax and after the projection layer."""
    @classmethod
    def make_local(cls) -> "AttentionConfig":
        """Create a small config for local use. Taken from nanogpt's local BabyGPT."""
        return cls(128, 32, 64, 4, 0)

    @classmethod
    def make_smoke(cls) -> "AttentionConfig":
        """Create a smoke test configuration."""
        return cls(8, 4, 16, 2, 0.2)


class ParallelMultiHeadAttention(nn.Module):
    """
    This class takes num of heads as a dimension and parallelizes the  self-attention computation.

    First, we take some input with dims [B, L, I] and shape it as [B, L, H, K, I]. Where B is the batch size, L is the
    sequence length, I is the input dimension, H is the number of heads and K=3 is the {key,value,query} triplet. Then,
    we evaluate the B batches in parallel for each head.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        sequence_length: int,
        num_heads: int,
        dropout: float = 0,
        is_causal: bool = True,
    ):
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

        self.dropout_att = nn.Dropout(dropout)
        self.dropout_project = nn.Dropout(dropout)

        self.use_flash = _use_flash_attention()
        self.dropout = dropout
        self.is_causal = is_causal

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

    def _mask_attention(self, att: torch.Tensor, seq_len: int) -> torch.Tensor:
        cropped_mask = self._att_mask[:, :, :seq_len, :seq_len]
        return att.masked_fill(cropped_mask, float("-inf"))

    def forward(self, x: torch.Tensor, att_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        The forward does not implement the residual inside -- following the identity highway idea.

        See class classname:`TransformerBlock`
        """
        # expected input dimensionality is [B, L, I]
        assert x.ndim == 3, f"Expected number of input dimensions to be 3, got {x.ndim}"
        bs, in_seq_len, _ = x.shape    # [B, L, I]

        x = self.qvk_linear(x)    # [B, L, KxHxh]
        q, k, v = self._broadcast_qvk(x)    # [B, H, L, h] x K

        if self.use_flash:
            att = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0,
                is_causal=self.is_causal,
                scale=self.scale,
            )
        else:
            # using inneficient manual implementation, following nanogpt
            # do (q [B, H , L, h] x k.T [B, H, h, L]) = qk [B, H, L, L]
            att = self.scale * (q @ k.transpose(-2, -1))
            if self.is_causal:
                att = self._mask_attention(att, in_seq_len)
            att = F.softmax(att, dim=-1)
            att = self.dropout_att(att)
            att = att @ v    # att [B, H, L, L] x v [B, H, L, h] = att [B, H, L, h]
        # concat output of attention heads, [B, L, Hxh]
        y = att.transpose(1, 2).contiguous().view(bs, in_seq_len, self.hidden_dim * self.num_heads)

        # project the output
        out = self.dropout_project(self.project(y))    # [B, L, I]
        # apply attention for padding
        return out

    @classmethod
    def from_config(cls, config: AttentionConfig, is_causal: bool = True) -> "ParallelMultiHeadAttention":
        return cls(
            config.input_dim, config.hidden_dim, config.sequence_length, config.num_heads, config.dropout, is_causal
        )
