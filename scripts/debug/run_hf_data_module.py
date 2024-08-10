from nanogpt.training.dataloader_fn import make_batches_fn
from nanogpt.training.datamodules.datamodules_utils import HFDatasetSpec
from nanogpt.training.datamodules.hf_data_module import HFDataModule
from nanogpt.training.tokenizer import HuggingFaceTokenizer

if __name__ == "__main__":
    dataset_spec = HFDatasetSpec(
        dataset_name="stas/openwebtext-10k", feature_name="text", valid_split_label=None, test_split_label=None
    )
    make_batches_fn = make_batches_fn(block_size=16, pad_token=-1)
    datamodule = HFDataModule(
        batch_size=64,
        tokenizer=HuggingFaceTokenizer(tokenizer_name="gpt2"),
        dataset_spec=dataset_spec,
        make_batches_fn=make_batches_fn,
    )
    datamodule.prepare_data()
    datamodule.setup("test")

    train_loader = datamodule.test_dataloader()
    print(next(iter(train_loader)))
