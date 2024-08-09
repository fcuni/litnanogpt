import os
from dataclasses import dataclass
from typing import Callable

import lightning as pl
from datasets import Dataset, DatasetDict, load_dataset, load_dataset_builder
from torch.utils.data import DataLoader, TensorDataset

from nanogpt.training.dataloader_fn import InputBatch, InputTokenized
from nanogpt.training.tokenizer import BaseTokenizer

# Number of cpus to use for data loading
N_WORKERS = os.cpu_count() // 2    # type: ignore
MakeBatchesFn = Callable[[InputTokenized], InputBatch]


@dataclass
class HFDatasetSpec:
    dataset_name: str
    """Name of the dataset to load as given in HF hub."""
    columns: list[str] | None = None
    """Columns to load from the dataset. If None, all columns are loaded."""
    feature_name: str | None = None
    """Name of the key in the dataset dict that refers to the text to use as data."""
    train_split_label: str = "train"
    """Label for the training split."""
    valid_split_label: str | None = "validation"
    """Label for the validation split. If None, we assume no validation split exist and use `train_test_split` to split the training data."""
    test_split_label: str | None = "test"
    """Label for the test split. If None, we assume no test split exist and use `test_split` to split the training data."""
    valid_split: float = 0.1
    """Fraction of the training data to use for testing if no validation split is provided."""
    test_split: float = 0.1
    """Fraction of the training data to use for testing if no test split is provided."""
    trust_remote_code: bool = False
    """Whether to trust remote code. If False and the dataset requires code execution, the instation will fail."""


class HFDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        tokenizer: BaseTokenizer,
        dataset_spec: HFDatasetSpec,
        make_batches_fn: MakeBatchesFn,
        data_dir: str = "data",
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
        self._seed = seed
        self._check_dataset_can_load()

        # predefine the train, validation and test datasets
        self.train_data: Dataset | None = None
        self.valid_data: Dataset | None = None
        self.test_data: Dataset | None = None

    def _check_dataset_can_load(self):
        print(f"Checking if dataset {self._dataset_name} can be loaded...")
        try:
            load_dataset_builder(
                self._dataset_name, trust_remote_code=self._dataset_spec.trust_remote_code
            )    # type: ignore
        except ValueError as e:
            raise ValueError(f"Could not find dataset {self._dataset_name} in HF datasets. {e}")
        print("Checks passed...")

    def prepare_data(self):
        pl.seed_everything(self._seed)
        # Set env variable to avoid deadlock issues with tokenizers
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if not os.path.isdir(self._dataset_path):
            os.makedirs(self._dataset_path)

        is_empty_dir = len(os.listdir(self._dataset_path)) == 0
        if is_empty_dir:
            print(f"Downloading dataset {self._dataset_name} to {self._dataset_path}...")
            load_dataset(
                self._dataset_name,
                num_proc=N_WORKERS,
                cache_dir=self._dataset_path,
                trust_remote_code=self._dataset_spec.trust_remote_code
            )

        else:
            print(f"Dataset {self._dataset_name} already exists at {self._dataset_path}, not redownloading.")

    def _tokenize(self, text_batch: dict):
        feature_name = self._dataset_spec.feature_name or list(text_batch.keys())[0]
        text = text_batch if isinstance(text_batch, str) else text_batch[feature_name]
        encoded_text = self._tokenizer.encode(text)
        return self._make_batches_fn(encoded_text)

    def setup(self, stage: str | None = None):
        dataset: DatasetDict = load_dataset(
            self._dataset_name,
            num_proc=N_WORKERS,
            cache_dir=self._dataset_path,
            trust_remote_code=self._dataset_spec.trust_remote_code
        )    # type: ignore
        train_data = dataset[self._dataset_spec.train_split_label]

        def _map(data: Dataset):
            # map into torch tensors of max len block_size and pad if necessary
            # return type is an iterator with InputBatch dicts
            return data.map(self._tokenize, batched=True, batch_size=None)

        if stage == "fit" or not stage:
            has_valid = self._dataset_spec.valid_split_label is not None
            if has_valid:
                _train_data = train_data
                valid_data = dataset[self._dataset_spec.valid_split_label]
            else:
                data_dict = train_data.train_test_split(test_size=self._dataset_spec.test_split, seed=self._seed)
                data_dict = data_dict["train"].train_test_split(
                    test_size=self._dataset_spec.valid_split, seed=self._seed
                )
                _train_data = data_dict["train"]
                valid_data = data_dict["test"]
                del data_dict

            self.train_data = _map(_train_data)
            self.train_data.set_format("torch", columns=self._dataset_spec.columns)

            self.valid_data = _map(valid_data)
            self.valid_data.set_format("torch", columns=self._dataset_spec.columns)
            del _train_data
            del valid_data

        if stage == "test" or not stage:
            has_test = self._dataset_spec.test_split_label is not None
            if has_test:
                test_data = dataset[self._dataset_spec.test_split_label]
            else:
                data_dict = train_data.train_test_split(test_size=self._dataset_spec.test_split, seed=self._seed)
                test_data = data_dict["test"]
                del data_dict

            self.test_data = _map(test_data)
            self.test_data.set_format("torch", columns=self._dataset_spec.columns)
            del test_data

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
