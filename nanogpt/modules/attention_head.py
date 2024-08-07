"""Initial (dumb) implementation. For the performant version see parallel_multi_head_attention.py."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = hidden_dim**-0.5

    def forward(self, x: torch.Tensor):
        # dims:
        #   x = [B, I]
        q = self.query(x)    #  [B, H]
        k = self.key(x)    #  [B, H]
        v = self.value(x)    #  [B, H]

        qk = F.softmax(self.scale * (q @ k.T), dim=-1)    #  [B, B]
        out = qk @ v    #  [B, H]
        return out


class MultiheadAttention(nn.Module):
    """Dumb non-parallelized multihead attention"""
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(input_dim, hidden_dim) for _ in range(num_heads)])
        self.output = nn.Linear(hidden_dim * num_heads, input_dim)

    def forward(self, x: torch.Tensor):
        # dims:
        #   x = [B, I]
        out = torch.cat([head(x) for head in self.heads], dim=-1)    #  [B, H * N]
        return self.output(out)    #  [B, I]
