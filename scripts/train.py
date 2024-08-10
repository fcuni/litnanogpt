import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from nanogpt.nanogpt_model import NanoGPT, NanoGPTConfig
from nanogpt.presets.preset_dataset_spec import get_openwebtxt_10k_spec
from nanogpt.training.dataloader_fn import make_batches_fn
from nanogpt.training.datamodules.hf_data_module import HFDataModule
from nanogpt.training.tokenizer import HuggingFaceTokenizer


def _use_wandb() -> bool:
    try:
        import wandb
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    dataset_spec = get_openwebtxt_10k_spec()
    tokenizer = HuggingFaceTokenizer(tokenizer_name="gpt2")

    conf = NanoGPTConfig.make_local()
    vocab_size = tokenizer.vocab_size
    conf.vocabulary_size = vocab_size or conf.vocabulary_size

    make_batches_fn = make_batches_fn(
        block_size=conf.attention_config.sequence_length, pad_token=tokenizer.pad_token_id
    )

    data = HFDataModule(batch_size=64, tokenizer=tokenizer, dataset_spec=dataset_spec, make_batches_fn=make_batches_fn)

    model = NanoGPT(config=conf)
    if _use_wandb():
        logger = WandbLogger(project="nanogpt", mode="online", save_code=True, log_model=True)
        logger.watch(model, log="all")
    else:
        logger = TensorBoardLogger("lightning_logs")
    callbacks = [ModelCheckpoint(monitor="valid/loss", save_top_k=1, mode="min")]

    trainer = pl.Trainer(
        max_epochs=100,
        log_every_n_steps=1,
        logger=logger,
        precision="16-mixed",
        callbacks=callbacks,    # type: ignore
        default_root_dir="checkpoints",
    )
    trainer.fit(model, datamodule=data)
