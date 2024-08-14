import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from nanogpt.nanogpt_model import NanoGPT, NanoGPTConfig
from nanogpt.presets.preset_dataset_spec import get_tiny_shakespeare_spec
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
    tokenizer = CharTokenizer()

    smoke_config = NanoGPTConfig.make_smoke()
    vocab_size = tokenizer.vocab_size
    smoke_config.vocabulary_size = vocab_size or smoke_config.vocabulary_size

    data = HFDataModule(
        batch_size=64,
        block_size=smoke_config.attention_config.sequence_length,
        tokenizer=tokenizer,
        dataset_spec=dataset_spec
    )

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
