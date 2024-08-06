from abc import abstractmethod

import torch
from torch import nn


class PositionalEncoder(nn.Module):
    def __init__(self, seq_len: int, emb_dim: int):
        super().__init__()
        self._seq_len = seq_len
        self._emb_dim = emb_dim

    @abstractmethod
    def forward(self, idxs: torch.Tensor) -> torch.Tensor:
        # input are the indices for the embedding matrix
        raise NotImplementedError


class VanillaPositionalEncoder(PositionalEncoder):
    def forward(self, idxs: torch.Tensor) -> torch.Tensor:
        # input are the indices for the embedding matrix
        return idxs / self._seq_len


class LearnedPositionalEncoder(PositionalEncoder):
    def __init__(self, seq_len: int, emb_dim: int):
        super().__init__(seq_len, emb_dim)
        self._embedding = nn.Embedding(seq_len, emb_dim)

    def forward(self, idxs: torch.Tensor) -> torch.Tensor:
        # input are the indices for the embedding matrix
        assert idxs.ndim == 1, \
        f"Expected a single dimension for the input tensor, representing the indices in the embedding, but got {idxs}"

        return self._embedding(idxs)
