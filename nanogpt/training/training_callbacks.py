import math

import lightning as pl
from lightning.pytorch.callbacks import Callback


class CosineLRCallback(Callback):
    def __init__(self, cosine_coefficient: float = 0.5, warmup_epochs: int = 5):
        self.cosine_coefficient = cosine_coefficient
        self.warmup_epochs = warmup_epochs - 1    # index starts at 0
        self.starting_lr = None

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        current_epoch = trainer.current_epoch
        if self.starting_lr is None:
            self.starting_lr = trainer.optimizers[0].param_groups[0]["lr"]
        if current_epoch < self.warmup_epochs:
            return

        # Cosine annealing
        cosine_f = (current_epoch - self.warmup_epochs) / (trainer.max_epochs - self.warmup_epochs)
        lr = self.starting_lr * self.cosine_coefficient * (1 + math.cos(math.pi * cosine_f))

        for param_group in trainer.optimizers[0].param_groups:
            param_group["lr"] = lr

        pl_module.log("train/lr", lr, on_step=False, on_epoch=True)
