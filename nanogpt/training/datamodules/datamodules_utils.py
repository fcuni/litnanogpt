import os
from dataclasses import dataclass
from typing import Callable

from nanogpt.training.dataloader_fn import InputBatch, InputTokenized

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
