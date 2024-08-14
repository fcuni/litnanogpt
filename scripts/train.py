import argparse

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from nanogpt.modules.positional_encoder import LearnedPositionalEncoder
from nanogpt.nanogpt_model import NanoGPT, NanoGPTConfig
from nanogpt.presets.preset_dataset_spec import get_spec_by_name
from nanogpt.training.datamodules.hf_data_module import HFDataModule
from nanogpt.training.tokenizer import CharTokenizer, HuggingFaceTokenizer
from nanogpt.training.training_callbacks import CosineLRCallback


def _use_wandb() -> bool:
    try:
        import wandb
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tokenizer", type=str, default="char", help="Tokenizer to use")
    parser.add_argument("-l", "--sequence_length", type=int, default=512, help="Sequence length for the model")
    parser.add_argument("-b", "--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("-d", "--dataset", type=str, default="tiny_shakespeare", help="Dataset name")
    parser.add_argument("-p", "--project", type=str, default="nanogpt", help="Project name for wandb")
    parser.add_argument("-m", "--mode", type=str, default="online", help="Mode for wandb")
    parser.add_argument("-n", "--notes", type=str, default="", help="Notes for wandb")
    parser.add_argument("-e", "--device", type=str, default="cuda", help="Device to train on")

    args = parser.parse_args()

    dataset_spec = get_spec_by_name(args.dataset)
    if args.tokenizer == "char":
        with open("data/tiny_shakespeare_vocab.pkl", "rb") as f:
            import pickle
            vocab = pickle.load(f)
        tokenizer = CharTokenizer(vocab=vocab)
    else:
        tokenizer = HuggingFaceTokenizer(tokenizer_name=args.tokenizer)

    conf = NanoGPTConfig.make_local()
    conf.positional_encoder = LearnedPositionalEncoder
    conf.vocabulary_size = tokenizer.vocab_size
    conf.attention_config.sequence_length = args.sequence_length

    data = HFDataModule(
        batch_size=args.batch_size, block_size=args.sequence_length, tokenizer=tokenizer, dataset_spec=dataset_spec
    )

    model = NanoGPT(config=conf)
    if _use_wandb():
        logger = WandbLogger(
            project=args.project,
            mode=args.mode,
            save_code=True,
            log_model=True,
            notes=args.notes if args.notes else None
        )
        logger.watch(model, log="all")
    else:
        logger = TensorBoardLogger("lightning_logs")
    callbacks = [ModelCheckpoint(monitor="valid/loss", save_top_k=1, mode="min"), CosineLRCallback()]

    trainer = pl.Trainer(
        max_epochs=10,
        log_every_n_steps=1,
        logger=logger,
        precision="16-mixed",
        callbacks=callbacks,    # type: ignore
        default_root_dir="checkpoints",
        gradient_clip_val=1.0,
        accelerator=args.device,
    )
    trainer.fit(model, datamodule=data)
