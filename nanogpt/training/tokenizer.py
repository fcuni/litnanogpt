from abc import abstractmethod

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES
from transformers.tokenization_utils_base import BatchEncoding

from nanogpt.training.dataloader_fn import InputTokenized


class BaseTokenizer:
    def __init__(self, pad_token_id: int | None = None, vocab_size: int | None = None):
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size

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
        super().__init__(pad_token_id=self._tokenizer.pad_token_id, vocab_size=len(self._tokenizer))

    def _make_tokenizer_from_name(self, tokenizer_name: str) -> PreTrainedTokenizer:
        try:
            return AutoTokenizer.from_pretrained(tokenizer_name)
        except OSError as e:
            raise OSError(
                f"Could not find tokenizer {tokenizer_name} in HF models. Try one of \n {TOKENIZER_MAPPING_NAMES} \n{e}"
            )

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
    """Tokenizer that encodes text as ASCII characters. Very simple and not recommended for real use."""
    def __init__(self, seq_len: int, vocab_size: int = 256):
        super().__init__(pad_token_id=vocab_size, vocab_size=vocab_size)
        self.oov_token = vocab_size + 1
        self.seq_len = seq_len
        # make space for oov and padding tokens
        self.vocab_size = vocab_size + 2

    def _pad_or_truncate(self, tokens: torch.Tensor) -> torch.Tensor:
        assert tokens.ndim == 1, f"Tokens must be 1D, got {tokens.shape}"
        tokens_len = tokens.size(0)
        if tokens_len < self.seq_len:
            pad_len = self.seq_len - len(tokens)
            padding = torch.tensor([self.pad_token_id] * pad_len)
            tokens = torch.cat([tokens, padding])
        elif tokens_len > self.seq_len:
            tokens = tokens[-self.seq_len:]
        return tokens

    def encode(self, text: list[str]) -> InputTokenized:
        tokens = []
        for s in text:
            tokens.append(self._pad_or_truncate(torch.tensor([ord(c) for c in s])))
        tokens = torch.stack(tokens)
        mask_ = torch.ones_like(tokens)
        mask_[tokens >= self.vocab_size] = 0
        tokens[mask_ == 0] = self.oov_token
        input_batch = InputTokenized(text=text, input_ids=tokens)
        return input_batch

    def decode(self, tokens: torch.Tensor) -> list[str]:
        sentences = []
        for i in range(tokens.size(0)):
            s = ""
            for c in tokens[i, :]:
                if c == self.pad_token_id:
                    break
                elif c == self.oov_token:
                    s += "<unk>"
                else:
                    s += chr(int(c))
            sentences.append(s)
        return sentences
