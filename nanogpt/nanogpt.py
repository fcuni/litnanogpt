from dataclasses import dataclass
from typing import Any, Callable

import lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import metric

from nanogpt.modules.parallel_multi_head_attention import AttentionConfig
from nanogpt.modules.positional_encoder import (PositionalEncoder, VanillaPositionalEncoder)
from nanogpt.modules.transformer_block import DecoderBlock

INPUT_BATCH = tuple[torch.Tensor, torch.Tensor]    # the tuple (input, labels)


@dataclass
class NanoGPTConfig:
    attention_config: AttentionConfig = AttentionConfig()
    ff_dims: list[int] = [128, 128]
    num_blocks: int = 3
    vocabulary_size: int = 50304    # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    positional_encoder: Callable[[int, int], PositionalEncoder] = VanillaPositionalEncoder

    @classmethod
    def make_smoke(cls) -> "NanoGPTConfig":
        """Create a smoke test configuration. Taken from nanogpt's BabyGPT."""
        smoke_att_config = AttentionConfig.make_smoke()
        ff_h_dim = [smoke_att_config.input_dim * 4, smoke_att_config.input_dim]
        return cls(smoke_att_config, ff_h_dim, 6)


class NanoGPT(pl.LightningModule):
    def __init__(self, config: NanoGPTConfig = NanoGPTConfig()):
        super().__init__()

        self.config = config
        att_config = config.attention_config
        self._block_length = att_config.sequence_length
        self.vocab_encoder = nn.Embedding(config.vocabulary_size, att_config.input_dim)
        self.positional_encoder = config.positional_encoder(att_config.sequence_length, att_config.input_dim)
        self._dropout = nn.Dropout(att_config.dropout)
        self.decoder = DecoderBlock(config.attention_config, config.ff_dims, config.num_blocks)
        # at this point, the output of the decoder is [B, L, I], the last layer produces [B, L, V], where V is the vocabulary_size
        self._layer_norm = nn.LayerNorm(att_config.input_dim)
        self.linear_project = nn.Linear(in_features=att_config.input_dim, out_features=config.vocabulary_size)

        # weight tying, see https://paperswithcode.com/method/weight-tying
        self.vocab_encoder.weight = self.linear_project.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # expected input shape is [B, L]
        _, l = x.shape
        assert l <= self._block_length, \
        f"The input length needs to be shorter than the max sequence_length, but got {l} and {self._block_length=}"

        encoder_idx = torch.arange(0, l, device=x.device, dtype=torch.long)
        positional_encoding = self.positional_encoder(encoder_idx)    # [L, I]

        vocab_encoding = self.vocab_encoder(x)    # [B, L, I]

        x_ = self._dropout(vocab_encoding + positional_encoding)    # [B, L, I]
        x_ = self.decoder(x_)    # [B, L, I]
        logits = self.linear_project(self._layer_norm(x_))    # [B, L, V]

        return logits

    def training_step(self, batch: INPUT_BATCH, batch_idx: int, step_type: str = "train"):
        # input at this point is a torch tensor of size [B, L], where each scalar doublet (b,l) corresponds to a token
        # index in the vocab, i.e. words have been tokenised and the output is some tensor
        # [[token_at_0, token_at_1, token_at_L], ...]
        x, labels = batch
        preds = self(x)
        loss = F.cross_entropy(preds, labels)
        metric_name = f"{step_type}/loss"
        if step_type == "test":
            return preds
        is_on_step = step_type == "train"
        self.log_dict({metric_name: loss}, logger=True, on_step=is_on_step, on_epoch=True)
        return loss

    def validation_step(self, batch: INPUT_BATCH, batch_idx: int):
        return self.training_step(batch, batch_idx, step_type="valid")

    def test_step(self, batch: INPUT_BATCH, batch_idx: int):
        return self.training_step(batch, batch_idx, step_type="test")

    def predict_step(
        self, batch: torch.Tensor, max_tokens: int, temperature: float = 1.0, top_k: int | None = None
    ) -> torch.Tensor:
        """Taken from the nanogpt, see `generate` in nanoGPT/model.py"""
        # expected input dims are [B, L], corresponding to b sentences of each length L tokens
        for _ in range(max_tokens):
            context_window = batch if batch.shape[-1] <= self._block_length else batch[:, -self._block_length:]

            pred_logits = self(context_window)[:, -1, :]    # take only last pred in the sequence
            pred_logits /= temperature

            if top_k is not None:
                assert top_k > 0, f"Expected a positive integer for the top k search, got {top_k}"
                top_k = min(top_k, pred_logits.size(-1))
                v, _ = torch.topk(pred_logits, top_k)
                mask_ = pred_logits < v[:, [-1]]
                pred_logits[mask_] = -float("inf")

            pred_probs = F.softmax(pred_logits, dim=-1)

            next_token = torch.multinomial(pred_probs, num_samples=1)
            batch = torch.cat([batch, next_token], dim=1)

        return batch

    def generate(
        self, batch: torch.Tensor, max_tokens: int, temperature: float = 1.0, top_k: int | None = None
    ) -> torch.Tensor:
        """Simple alias for predict_step. More clear semantics outside the Lightning Trainer API."""
        return self.predict_step(batch, max_tokens, temperature, top_k)

    def configure_optimizers(self):
        ...
