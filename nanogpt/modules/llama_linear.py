import torch
from torch import nn


class LlamaLinear(nn.Module):
    """
    Linear module found in the Llama-like models.
    """
    def __init__(
        self,
        in_dimensions: int,
        hidden_dimensions: int | None = None,
        out_dimensions: int | None = None,
        bias: bool = True
    ):
        super().__init__()
        hidden_dimensions = hidden_dimensions or 4 * in_dimensions
        out_dimensions = out_dimensions or in_dimensions

        self._up = nn.Linear(in_features=in_dimensions, out_features=hidden_dimensions, bias=bias)
        self._down = nn.Linear(in_features=hidden_dimensions, out_features=out_dimensions, bias=bias)
        self._gate = nn.Linear(in_features=in_dimensions, out_features=hidden_dimensions, bias=bias)

        self._silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._silu(self._gate(x)) * self._up(x)
        return self._down(x)
