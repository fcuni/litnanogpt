import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from nanogpt.nanogpt_model import NanoGPT, NanoGPTConfig
from nanogpt.presets.preset_dataset_spec import get_tiny_shakespeare_spec
from nanogpt.training.dataloader_fn import make_batches_fn
from nanogpt.training.datamodules.hf_data_module import HFDataModule
from nanogpt.training.tokenizer import CharTokenizer, HuggingFaceTokenizer


def _use_wandb() -> bool:
    try:
        import wandb
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    dataset_spec = get_tiny_shakespeare_spec()
    # tokenizer = HuggingFaceTokenizer(tokenizer_name="gpt2")
    tokenizer = CharTokenizer(seq_len=16)

    smoke_config = NanoGPTConfig.make_smoke()
    vocab_size = tokenizer.vocab_size
    smoke_config.vocabulary_size = vocab_size or smoke_config.vocabulary_size

    make_batches_fn = make_batches_fn(
        block_size=smoke_config.attention_config.sequence_length, pad_token=tokenizer.pad_token_id
    )

    data = HFDataModule(batch_size=64, tokenizer=tokenizer, dataset_spec=dataset_spec, make_batches_fn=make_batches_fn)

    logger = WandbLogger(project="nanogpt", mode="disabled") if _use_wandb() else TensorBoardLogger("lightning_logs")

    model = NanoGPT(config=smoke_config)
    trainer = pl.Trainer(
        max_epochs=1,
        log_every_n_steps=1,
        logger=logger,
        precision="16-mixed",
        enable_checkpointing=False,
        accelerator="cpu"
    )
    trainer.fit(model, datamodule=data)
