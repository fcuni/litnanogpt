import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from nanogpt.nanogpt_model import NanoGPT, NanoGPTConfig
from nanogpt.training.dataloader_fn import make_batches_fn
from nanogpt.training.hf_data_module import HFDataModule, HFDatasetSpec
from nanogpt.training.tokenizer import HuggingFaceTokenizer


def _use_wandb() -> bool:
    try:
        import wandb
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    smoke_config = NanoGPTConfig.make_smoke()
    model = NanoGPT(config=smoke_config)

    dataset_spec = HFDatasetSpec(dataset_name="karpathy/tiny_shakespeare", feature_name="text", trust_remote_code=True)

    tokenizer = HuggingFaceTokenizer(tokenizer_name="gpt2")
    vocab_size = tokenizer.vocab_size
    smoke_config.vocabulary_size = vocab_size or smoke_config.vocabulary_size

    make_batches_fn = make_batches_fn(
        block_size=smoke_config.attention_config.sequence_length,
        vocab_size=smoke_config.vocabulary_size,
        pad_token=tokenizer.pad_token_id
    )

    data = HFDataModule(
        batch_size=64,
        block_size=smoke_config.attention_config.sequence_length,
        tokenizer=tokenizer,
        dataset_spec=dataset_spec,
        make_batches_fn=make_batches_fn
    )

    logger = WandbLogger(project="nanogpt", mode="online") if _use_wandb() else TensorBoardLogger("lightning_logs")

    trainer = pl.Trainer(
        max_epochs=10,
        log_every_n_steps=1,
        logger=logger,
        precision="16-mixed",
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=data)
