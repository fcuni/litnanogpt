from nanogpt.training.hf_data_module import HFDataModule, HFDatasetSpec
from nanogpt.training.tokenizer import HuggingFaceTokenizer

if __name__ == "__main__":
    dataset_spec = HFDatasetSpec(
        dataset_name="stas/openwebtext-10k", feature_name="text", valid_split_label=None, test_split_label=None
    )
    datamodule = HFDataModule(
        batch_size=64,
        tokenizer=HuggingFaceTokenizer(tokenizer_name="gpt2"),
        dataset_spec=dataset_spec,
        block_size=256,
    )
    datamodule.prepare_data()
    datamodule.setup("test")

    train_loader = datamodule.test_dataloader()
    print(next(iter(train_loader)))
    __import__('pdb').set_trace()
