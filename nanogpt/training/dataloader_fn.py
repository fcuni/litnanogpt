from typing import Callable, NotRequired, TypedDict

import torch


class InputTokenized(TypedDict):
    text: str
    input_ids: torch.Tensor
    attention_mask: NotRequired[torch.Tensor]


class InputBatch(TypedDict):
    input: torch.Tensor
    labels: torch.Tensor
    attention_mask: NotRequired[torch.Tensor]


def make_batches_fn(block_size: int, pad_token: int | None = None) -> Callable[[InputTokenized], InputBatch]:
    pad_token = pad_token or -1

    def _make_batches(encoding: InputTokenized) -> InputBatch:
        input_ = encoding["input_ids"]
        seq_len = input_.size(1)
        pad_last_position = torch.tensor([pad_token]).expand(input_.size(0), 1)
        labels = torch.cat([input_[:, 1:], pad_last_position], dim=1)

        if encoding.get("attention_mask") is not None:
            attention_mask = encoding["attention_mask"]    # type: ignore
        else:
            attention_mask = torch.ones_like(input_)

        if seq_len > block_size:
            input_ = input_[:, :block_size]
            labels = labels[:, :block_size]
            attention_mask = attention_mask[:, :block_size]

        return {"input": input_, "labels": labels, "attention_mask": attention_mask}

    return _make_batches
