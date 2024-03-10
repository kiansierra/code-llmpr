from typing import List, Optional
from datasets import load_dataset, Dataset, concatenate_datasets

NEW_KEY_COLUMN = "original_text"

__all__ = ["dataset_preprocess"]

def dataset_preprocess(dataset_name: str, key_column: str ,subsets:Optional[List[str]] = None) -> Dataset:
    subsets = subsets or ["train"]
    dataset = load_dataset(dataset_name)
    dataset = dataset.rename_column(key_column, NEW_KEY_COLUMN)
    dataset = dataset.select_columns([NEW_KEY_COLUMN])
    dataset = concatenate_datasets([dataset[subset] for subset in subsets])
    return dataset