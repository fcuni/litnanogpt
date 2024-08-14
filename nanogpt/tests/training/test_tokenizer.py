import pytest

from nanogpt.training.tokenizer import CharTokenizer, HuggingFaceTokenizer


@pytest.mark.parametrize("phrase", [["hello"], ["hello world"]])
def test_char_tokenizer_returns_single_character_tokens(phrase):
    tokenizer = CharTokenizer()
    tokenized_input = tokenizer.encode(phrase)
    tokens = tokenized_input.get("input_ids")

    pad_token_id = tokenizer.pad_token_id
    tokens = tokens[tokens != pad_token_id].unsqueeze(0)
    sentence_decoded = tokenizer.decode(tokens)
    assert sentence_decoded == phrase, f"Expected {phrase=}, got {sentence_decoded=}"


def test_char_tokenizer_handles_oov_tokens():
    tokenizer = CharTokenizer(vocab={"a": 1})
    input_text = ["h"]
    tokenized_input = tokenizer.encode(input_text)
    tokens = tokenized_input.get("input_ids")

    oov_token = tokenizer.oov_token
    assert tokens[0, 0] == oov_token, f"Expected {oov_token=}, got {tokens[0, 0]}"


def test_char_tokenizer_uses_custom_vocab():
    vocab = {"a": 1, "b": 2}
    tokenizer = CharTokenizer(vocab=vocab)
    input_text = ["a", "b"]
    tokenized_input = tokenizer.encode(input_text)
    tokens = tokenized_input.get("input_ids")
    assert tokens[0, 0] == vocab["a"], f"Expected {vocab['a']=}, got {tokens[0, 0]}"
    assert tokens[0, 1] == vocab["b"], f"Expected {vocab['b']=}, got {tokens[0, 1]}"


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
