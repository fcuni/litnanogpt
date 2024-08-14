import os
from typing import cast

import lightning as pl
from datasets import Dataset, DatasetDict, load_dataset, load_dataset_builder

from nanogpt.training.datamodules.base_data_module import BaseNLPDataModule
from nanogpt.training.datamodules.datamodules_utils import N_WORKERS, HFDatasetSpec
from nanogpt.training.tokenizer import BaseTokenizer


class HFDataModule(BaseNLPDataModule):
    def __init__(
        self,
        batch_size: int,
        block_size: int,
        tokenizer: BaseTokenizer,
        dataset_spec: HFDatasetSpec,
        data_dir: str = "data",
        n_workers: int = N_WORKERS,
        seed: int = 42,
    ):
        super().__init__(
            batch_size=batch_size,
            block_size=block_size,
            tokenizer=tokenizer,
            dataset_spec=dataset_spec,
            data_dir=data_dir,
            n_workers=n_workers,
            seed=seed,
        )

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
        text = [text_batch] if isinstance(text_batch, str) else text_batch[feature_name]
        text = cast(list[str], text)
        encoded_text = self._tokenizer.encode(text)
        batches = self._tokenizer.make_batches(encoded_text, self._block_size)
        return batches

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
            return data.map(self._tokenize, batched=True, batch_size=None, remove_columns=data.column_names)

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
