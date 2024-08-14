import pickle

import torch
from datasets import load_dataset

from nanogpt.presets.preset_dataset_spec import get_tiny_shakespeare_spec
from nanogpt.training.datamodules.hf_data_module import HFDataModule
from nanogpt.training.tokenizer import CharTokenizer

# the numbes in the main script are taken from https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/prepare.py
if __name__ == "__main__":
    tiny_shakespeare = get_tiny_shakespeare_spec()
    dataset = load_dataset(tiny_shakespeare.dataset_name, cache_dir="data", num_proc=4, trust_remote_code=True)
    train_data = dataset[tiny_shakespeare.train_split_label]    # type: ignore

    # make vocab from unique chars in train_data
    train_text = train_data["text"]    # type: ignore
    unique_chars = sorted(list(set("".join(train_text))))
    vocab = {char: i for i, char in enumerate(unique_chars)}

    # make sure we have the same vocab as karpathy
    assert len(vocab) == 65, f"Expected 65 unique chars, got {len(vocab)}"

    tokenizer = CharTokenizer(vocab=vocab)

    tokenized_input = tokenizer.encode(train_data[tiny_shakespeare.feature_name])    # type: ignore

    tokens = tokenized_input.get("input_ids")

    assert tokens.size(-1) == 1003854, f"Expected 1003854 tokens, got {len(tokens)}"

    features = tokens[:, :12 * 64].reshape(12, 64)
    labels = tokens[:, 1:12 * 64 + 1].reshape(12, 64)

    # these are extracted from karpathy batch loading
    with open("../nanoGPT/data/shakespeare_char/one_batch.pkl", "rb") as f:
        f, l = pickle.load(f)

    # compare with our generated batch from the encoder
    assert torch.allclose(features, f)
    assert torch.allclose(labels, l)

    # make sure the hf datamodule works correctly as well
    dm = HFDataModule(
        batch_size=12,
        block_size=64,
        tokenizer=tokenizer,
        dataset_spec=tiny_shakespeare,
    )

    dm.setup()
    train_loader = dm.train_dataloader(shuffle=False)

    first_batch = next(iter(train_loader))
    f_loader, l_loader = first_batch["input"], first_batch["labels"]

    assert torch.allclose(f_loader, f)
    assert torch.allclose(l_loader, l)
