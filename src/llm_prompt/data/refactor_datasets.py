from typing import List, Optional

from datasets import Dataset, concatenate_datasets, load_dataset
from omegaconf import DictConfig, OmegaConf

NEW_KEY_COLUMN = "original_text"

__all__ = ["dataset_preprocess"]


def dataset_preprocess(
    dataset_name: str,
    key_column: str,
    subset: Optional[str] = None,
    splits: Optional[List[str]] = None,
    data_files: Optional[str] = None,
) -> Dataset:
    splits = splits or ["train"]
    if data_files is None:
        datasets = load_dataset(dataset_name, name=subset)
    else:
        if isinstance(data_files, DictConfig):
            data_files = OmegaConf.to_container(data_files)
        datasets = load_dataset(dataset_name, data_files=data_files)
    datasets = datasets.rename_column(key_column, NEW_KEY_COLUMN)
    datasets = datasets.select_columns([NEW_KEY_COLUMN])
    dataset_list = []
    for subset in splits:
        dataset = datasets[subset]
        dataset = dataset.add_column("source", [dataset_name] * len(dataset))
        dataset = dataset.add_column("split", [subset] * len(dataset))
        dataset = dataset.map(lambda x: {"original_length": len(x.split(" "))}, input_columns=NEW_KEY_COLUMN)
        dataset_list.append(dataset)
    dataset = concatenate_datasets(dataset_list)
    return dataset
