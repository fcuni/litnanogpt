from abc import abstractmethod
from typing import TypedDict

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES
from transformers.tokenization_utils_base import BatchEncoding

PADDING_INDEX = 0


class InputTokenized(TypedDict):
    text: str | list[str]    # Double check this, it migth only ever be a list
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class InputBatch(TypedDict):
    input: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor


class BaseTokenizer:
    def __init__(self, pad_token_id: int | None = None, vocab_size: int | None = None):
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size

    @abstractmethod
    def make_batches(self, encoding: InputTokenized, sequence_length: int) -> InputBatch:
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: list[str]):
        raise NotImplementedError

    @abstractmethod
    def decode(self, tokens: torch.Tensor) -> list[str]:
        raise NotImplementedError


class HuggingFaceTokenizer(BaseTokenizer):
    def __init__(self, tokenizer_name: str):
        self._tokenizer = self._make_tokenizer_from_name(tokenizer_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self._tokenizer.pad_token_id = PADDING_INDEX
        super().__init__(pad_token_id=self._tokenizer.pad_token_id, vocab_size=len(self._tokenizer))

    def _make_tokenizer_from_name(self, tokenizer_name: str) -> PreTrainedTokenizer:
        try:
            return AutoTokenizer.from_pretrained(tokenizer_name)
        except OSError as e:
            raise OSError(
                f"Could not find tokenizer {tokenizer_name} in HF models. Try one of \n {TOKENIZER_MAPPING_NAMES} \n{e}"
            )

    def make_batches(self, encoding: InputTokenized, sequence_length: int) -> InputBatch:
        input_ = encoding["input_ids"]
        seq_len = input_.size(1)
        pad_last_position = torch.tensor([self.pad_token_id]).expand(input_.size(0), 1)
        labels = torch.cat([input_[:, 1:], pad_last_position], dim=1)
        if encoding.get("attention_mask") is not None:
            attention_mask = encoding["attention_mask"]
        else:
            attention_mask = torch.ones_like(input_)
        if seq_len > sequence_length:
            input_ = input_[:, :sequence_length]
            labels = labels[:, :sequence_length]
            attention_mask = attention_mask[:, :sequence_length]
        return {"input": input_, "labels": labels, "attention_mask": attention_mask}

    def encode(self, text: list[str]) -> BatchEncoding:
        return self._tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    def decode(self, tokens: torch.Tensor) -> list[str]:
        sentences = []
        for i in range(tokens.size(0)):
            sentences += [
                self._tokenizer.decode(tokens[i, :], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            ]
        return sentences


class CharTokenizer(BaseTokenizer):
    """Character-level tokenizer that mimicks Karpathy's character tokenizer. Used mostly to check for consistency."""
    def __init__(self, vocab: dict[str, int] | None = None):
        self.vocab = vocab or {chr(i): i - 31 for i in range(32, 126)}    # ASCII
        vocab_size = len(self.vocab)
        self.special_tokens = {"<unk>": -3, "<eot>": -2, "<pad>": PADDING_INDEX}
        self.oov_token = self.special_tokens["<unk>"]
        pad_token_id = self.special_tokens["<pad>"]
        self.extended_vocab = {**self.vocab, **self.special_tokens}
        self.inv_extended_vocab = {v: k for k, v in self.extended_vocab.items()}
        super().__init__(pad_token_id=pad_token_id, vocab_size=vocab_size)

    def make_batches(self, encoding: InputTokenized, sequence_length: int) -> InputBatch:
        input_ = encoding["input_ids"].squeeze(0)
        assert input_.dim() == 1, f"Expected input to be a 1D tensor, but got {input_.dim()}"
        # pad if needed
        pad_len = sequence_length - input_.size(0) % sequence_length

        if pad_len > 0:
            input_ = torch.cat([input_, torch.tensor([self.pad_token_id] * pad_len)])
        labels = torch.cat([input_[1:], torch.tensor([self.pad_token_id])]).reshape(-1, sequence_length)
        input_ = input_.reshape(-1, sequence_length)
        if encoding.get("attention_mask") is not None:
            attention_mask = encoding["attention_mask"].squeeze(0)    # type: ignore
            if pad_len > 0:
                attention_mask = torch.cat([attention_mask, torch.zeros(pad_len)])
            attention_mask = attention_mask.reshape(-1, sequence_length)
        else:
            attention_mask = torch.ones_like(input_)
        return {"input": input_, "labels": labels, "attention_mask": attention_mask}

    def encode(self, text: list[str]) -> InputTokenized:
        tokens, att_masks = [], []
        for s in text:
            t = torch.tensor([self.extended_vocab[c] if c in self.extended_vocab else self.oov_token for c in s])
            tokens.append(t)
        tokens = torch.cat(tokens, dim=0).unsqueeze(0)
        att_masks = torch.ones_like(tokens)
        input_batch = InputTokenized(text=text, input_ids=tokens, attention_mask=att_masks)
        return input_batch

    def decode(self, tokens: torch.Tensor) -> list[str]:
        sentences = []
        for i in range(tokens.size(0)):
            s = ""
            for c in tokens[i, :]:
                char = self.inv_extended_vocab[c.item()]
                if char == "<eot>":
                    break
                elif char == "<pad>":
                    continue
                s += char
            sentences.append(s)
        return sentences
