import os
from abc import abstractmethod

import lightning as pl
from datasets import Dataset
from torch.utils.data import DataLoader, TensorDataset

from nanogpt.training.datamodules.datamodules_utils import (N_WORKERS, HFDatasetSpec, MakeBatchesFn)
from nanogpt.training.tokenizer import BaseTokenizer


class BaseNLPDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        tokenizer: BaseTokenizer,
        dataset_spec: HFDatasetSpec,
        make_batches_fn: MakeBatchesFn,
        data_dir: str = "data",
        n_workers: int = N_WORKERS,
        seed: int = 42,
    ):
        super().__init__()
        self._batch_size = batch_size
        self._tokenizer = tokenizer
        self._data_dir = f"{os.getcwd()}/{data_dir}"
        self._dataset_name = dataset_spec.dataset_name
        self._dataset_path = f"{self._data_dir}/{self._dataset_name}"
        self._dataset_spec = dataset_spec
        self._make_batches_fn = make_batches_fn
        self._n_data_workers = n_workers
        self._seed = seed

        # predefine the train, validation and test datasets
        self.train_data: Dataset | None = None
        self.valid_data: Dataset | None = None
        self.test_data: Dataset | None = None

    @abstractmethod
    def prepare_data(self):
        raise NotImplementedError

    @abstractmethod
    def _tokenize(self, text_batch: dict):
        raise NotImplementedError

    @abstractmethod
    def setup(self, stage: str | None = None):
        raise NotImplementedError

    def _build_dataloader(self, dataset: Dataset, shuffle: bool = True):
        return DataLoader(
            dataset,    # type: ignore
            batch_size=self._batch_size,
            num_workers=N_WORKERS,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_data is not None, "Data not loaded. Call `setup` first."
        return self._build_dataloader(self.train_data)

    def val_dataloader(self) -> DataLoader:
        assert self.valid_data is not None, "Data not loaded. Call `setup` first."
        return self._build_dataloader(self.valid_data, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        assert self.test_data is not None, "Data not loaded. Call `setup` first."
        return self._build_dataloader(self.test_data, shuffle=False)

    def make_prediction_dataloader(self, text: str) -> DataLoader:
        tokenized_text = self._tokenizer.encode(text)
        dataset = TensorDataset(tokenized_text)
        return DataLoader(dataset, batch_size=1, num_workers=1)
