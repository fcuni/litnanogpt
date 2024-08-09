import torch
from torch import nn

from nanogpt.modules.parallel_multi_head_attention import (AttentionConfig, ParallelMultiHeadAttention)


class TransformerBlock(nn.Module):
    def __init__(self, attention_config: AttentionConfig = AttentionConfig(), ff_dims: list[int] | None = None):
        super().__init__()

        input_dim = attention_config.input_dim
        self._layer_norm_att = nn.LayerNorm(input_dim)
        self._layer_norm_projection = nn.LayerNorm(input_dim)

        self._mh_att = ParallelMultiHeadAttention.from_config(attention_config, is_causal=False)

        ff_dims = ff_dims or [4 * input_dim, input_dim]

        layers = []
        hidden_dim = input_dim
        for size in ff_dims:
            layers.append(nn.Linear(hidden_dim, size))
            hidden_dim = size
        self._ff_projection = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = self._layer_norm_att(x)
        x_ = self._mh_att(x_)

        x_ = x_ + x    # add residual before ff projection
        x_ = self._layer_norm_projection(x_)
        x_ = self._ff_projection(x_)

        # add residual before return to support identity highway
        return x_ + x


class EncoderBlock(nn.Module):
    def __init__(self, attention_config: AttentionConfig, ff_dims: list[int] | None = None, num_blocks: int = 3):
        super().__init__()

        self._transformer_blocks = nn.ModuleList([
            TransformerBlock(attention_config=attention_config, ff_dims=ff_dims) for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self._transformer_blocks:
            x = l(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, attention_config: AttentionConfig, ff_dims: list[int] | None = None, num_blocks: int = 3):
        super().__init__()

        masked_blocks = []
        for _ in range(num_blocks):
            masked_blocks += [
                nn.Sequential(
                    *[
                        nn.LayerNorm(attention_config.input_dim),
                        ParallelMultiHeadAttention.from_config(attention_config, is_causal=True),
                    ]
                )
            ]
        self._masked_blocks = nn.ModuleList(masked_blocks)
        self._transformer_blocks = nn.ModuleList([
            TransformerBlock(attention_config=attention_config, ff_dims=ff_dims) for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        x_ = x
        h = h or torch.zeros_like(x)
        for m, l in zip(self._masked_blocks, self._transformer_blocks):
            x_ = m(x_)
            x_ = x_ + h + x
            x_ = l(x_)
            x_ = x_ + x
        return x_


class EncoderDecoderBlock(nn.Module):
    def __init__(self, attention_config: AttentionConfig, ff_dims: list[int] | None = None, num_blocks: int = 3):
        super().__init__()

        self._encoder = EncoderBlock(attention_config, ff_dims, num_blocks)
        self._decoder = DecoderBlock(attention_config, ff_dims, num_blocks)

    def forward(self,
                x_input: torch.Tensor,
                x_output: torch.Tensor,
                h: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        h = h or self._encoder(x_input)
        x = self._decoder(x_output, h)
        return x, h    # type: ignore
