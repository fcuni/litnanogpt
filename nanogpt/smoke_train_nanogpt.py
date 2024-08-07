import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from nanogpt.nanogpt import NanoGPT, NanoGPTConfig


def _use_wandb() -> bool:
    try:
        import wandb
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    smoke_config = NanoGPTConfig.make_smoke()
    model = NanoGPT(config=smoke_config)

    data = ...
    logger = WandbLogger(project="nanogpt") if _use_wandb() else TensorBoardLogger("lightning_logs")

    trainer = pl.Trainer(
        max_epochs=1,
        log_every_n_steps=10,
        logger=logger,
    )
    trainer.fit(model, data)
