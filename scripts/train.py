import argparse

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from nanogpt.nanogpt_model import NanoGPT, NanoGPTConfig
from nanogpt.presets.preset_dataset_spec import get_openwebtxt_10k_spec
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tokenizer", type=str, default="char", help="Tokenizer to use")
    parser.add_argument("-l", "--seq_len", type=int, default=512, help="Sequence length for the model")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("-d", "--dataset", type=str, default="openwebtxt-10k", help="Dataset name")
    parser.add_argument("-p", "--project", type=str, default="nanogpt", help="Project name for wandb")
    parser.add_argument("-m", "--mode", type=str, default="online", help="Mode for wandb")

    args = parser.parse_args()

    dataset_spec = get_openwebtxt_10k_spec()
    if args.tokenizer == "char":
        tokenizer = CharTokenizer(seq_len=args.seq_len)
    else:
        tokenizer = HuggingFaceTokenizer(tokenizer_name=args.tokenizer)

    conf = NanoGPTConfig.make_local()
    vocab_size = tokenizer.vocab_size
    conf.vocabulary_size = vocab_size or conf.vocabulary_size
    conf.attention_config.sequence_length = args.seq_len

    make_batches_fn = make_batches_fn(
        block_size=conf.attention_config.sequence_length, pad_token=tokenizer.pad_token_id
    )

    data = HFDataModule(
        batch_size=args.batch_size, tokenizer=tokenizer, dataset_spec=dataset_spec, make_batches_fn=make_batches_fn
    )

    model = NanoGPT(config=conf)
    if _use_wandb():
        logger = WandbLogger(project=args.project, mode=args.mode, save_code=True, log_model=True)
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
