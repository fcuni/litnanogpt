import pytest

from nanogpt.training.tokenizer import CharTokenizer, HuggingFaceTokenizer


@pytest.mark.parametrize("seq_len, vocab_size, phrase", [(5, 256, "hello"), (10, 256, "hello")])
def test_char_tokenizer_returns_single_character_tokens(seq_len, vocab_size, phrase):
    tokenizer = CharTokenizer(seq_len=seq_len, vocab_size=vocab_size)
    tokens = tokenizer.encode(phrase)

    # expected encoding length is seq_len
    assert tokens.size(-1) == seq_len, f"Expected length to be {seq_len=}, got {tokens.size(-1)}"

    # TODO: move the remaninder of this test to a separate test
    # if we slice out the padding tokens, we should have the original phrase
    pad_token_id = tokenizer.pad_token_id
    tokens = tokens[tokens != pad_token_id].unsqueeze(0)
    sentence_decoded = "".join(tokenizer.decode(tokens))
    assert sentence_decoded == phrase, f"Expected {phrase=}, got {sentence_decoded=}"


def test_char_tokenizer_handles_oov_tokens():
    tokenizer = CharTokenizer(seq_len=5, vocab_size=80)
    input_text = "h"
    tokens = tokenizer.encode(input_text)

    oov_token = tokenizer.oov_token
    assert tokens[0, 0] == oov_token, f"Expected {oov_token=}, got {tokens[0, 0]}"


def test_huggingface_tokenizer_raises_error_with_bad_name():
    with pytest.raises(OSError):
        HuggingFaceTokenizer(tokenizer_name="this_name_does_not_exist")


@pytest.mark.parametrize("tokenizer_name", ["gpt2", "distilgpt2"])
def test_huggingface_tokenizer_encodes_and_decodes(tokenizer_name):
    tokenizer = HuggingFaceTokenizer(tokenizer_name=tokenizer_name)
    text = "hello world"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens["input_ids"])
    assert decoded == [text], f"Expected {text=}, got {decoded=}"
