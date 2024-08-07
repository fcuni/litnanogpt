from abc import abstractmethod

import tiktoken
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES
from transformers.tokenization_utils_base import BatchEncoding


class BaseTokenizer:
    @abstractmethod
    def encode(self, text: str):
        raise NotImplementedError

    @abstractmethod
    def decode(self, tokens: torch.Tensor) -> str:
        raise NotImplementedError


class TikTokenizer(BaseTokenizer):
    def __init__(self, encoding: str):
        self._check_encoding_is_valid(encoding)
        self._tokenizer = tiktoken.get_encoding(encoding)

    def _check_encoding_is_valid(self, encoding: str):
        encodings = tiktoken.list_encoding_names()
        if encoding not in encodings:
            raise ValueError(f"Could not find {encoding} in TikToken encodings, should be one of: {encodings}")

    def encode(self, text: str) -> dict[str, torch.Tensor]:
        tokens = torch.tensor(self._tokenizer.encode(text), dtype=torch.long)
        return {"tokens": tokens}

    def decode(self, tokens: torch.Tensor) -> str:
        token_list = tokens.tolist()
        return self._tokenizer.decode(token_list)


class HuggingFaceTokenizer(BaseTokenizer):
    def __init__(self, tokenizer_name: str):
        self._tokenizer = self._make_tokenizer_from_name(tokenizer_name)

    def _make_tokenizer_from_name(self, tokenizer_name: str) -> PreTrainedTokenizer:
        try:
            return AutoTokenizer.from_pretrained(tokenizer_name)
        except ValueError as e:
            raise ValueError(
                f"Could not find tokenizer {tokenizer_name} in HF models. Try one of \n {TOKENIZER_MAPPING_NAMES} \n{e}"
            )

    def encode(self, text: str) -> BatchEncoding:
        if self._tokenizer.pad_token is None:
            self._tokenizer.add_special_tokens({"pad_token": "<pad>"})
        return self._tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    def decode(self, tokens: torch.Tensor) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=True)


class LlamaTokenizer(BaseTokenizer):
    # TODO: Add tokenizer for Llama models
    ...
