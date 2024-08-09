from abc import abstractmethod

import tiktoken
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES
from transformers.tokenization_utils_base import BatchEncoding


class BaseTokenizer:
    def __init__(self):
        self.pad_token_id: int | None = None
        self.vocab_size: int | None = None

    @abstractmethod
    def encode(self, text: str):
        raise NotImplementedError

    @abstractmethod
    def decode(self, tokens: torch.Tensor) -> str:
        raise NotImplementedError


class HuggingFaceTokenizer(BaseTokenizer):
    def __init__(self, tokenizer_name: str):
        self._tokenizer = self._make_tokenizer_from_name(tokenizer_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.pad_token_id = self._tokenizer.pad_token_id
        self.vocab_size = len(self._tokenizer)

    def _make_tokenizer_from_name(self, tokenizer_name: str) -> PreTrainedTokenizer:
        try:
            return AutoTokenizer.from_pretrained(tokenizer_name)
        except ValueError as e:
            raise ValueError(
                f"Could not find tokenizer {tokenizer_name} in HF models. Try one of \n {TOKENIZER_MAPPING_NAMES} \n{e}"
            )

    def encode(self, text: str) -> BatchEncoding:
        return self._tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    def decode(self, tokens: torch.Tensor) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=True)


class LlamaTokenizer(BaseTokenizer):
    # TODO: Add tokenizer for Llama models
    ...
