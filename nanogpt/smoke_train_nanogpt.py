import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from nanogpt.nanogpt_model import NanoGPT, NanoGPTConfig
from nanogpt.training.data_module import DatasetOrigin, NLPDataModule
from nanogpt.training.tokenizer import TikTokenizer


def _use_wandb() -> bool:
    try:
        import wandb
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    smoke_config = NanoGPTConfig.make_smoke()
    model = NanoGPT(config=smoke_config)

    tokenizer = TikTokenizer(encoding="gpt2")
    data = NLPDataModule(
        batch_size=64, tokenizer=tokenizer, dataset_name="ptb_text_only", dataset_origin=DatasetOrigin.HUGGINGFACE
    )
    logger = WandbLogger(project="nanogpt", mode="disabled") if _use_wandb() else TensorBoardLogger("lightning_logs")

    trainer = pl.Trainer(
        max_epochs=1,
        log_every_n_steps=10,
        logger=logger,
    )
    trainer.fit(model, datamodule=data)
