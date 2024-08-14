from nanogpt.presets.preset_dataset_spec import get_spec_by_name
from nanogpt.training.datamodules.hf_data_module import HFDataModule
from nanogpt.training.tokenizer import CharTokenizer, HuggingFaceTokenizer

if __name__ == "__main__":
    dataset_spec = get_spec_by_name("tiny_shakespeare")
    tokenizer = CharTokenizer()
    # tokenizer = HuggingFaceTokenizer(tokenizer_name="gpt2")

    datamodule = HFDataModule(
        batch_size=64,
        block_size=64,
        tokenizer=tokenizer,
        dataset_spec=dataset_spec,
    )
    datamodule.prepare_data()
    datamodule.setup("fit")

    train_loader = datamodule.train_dataloader()
    sample = next(iter(train_loader))
    print(next(iter(train_loader)))
