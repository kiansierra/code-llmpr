from typing import List, Optional
from datasets import load_dataset, Dataset, concatenate_datasets

NEW_KEY_COLUMN = "original_text"

__all__ = ["dataset_preprocess", "REWRITE_PROMPTS", "REWRITE_TEMPLATE"]

def dataset_preprocess(dataset_name: str, key_column: str ,subsets:Optional[List[str]] = None) -> Dataset:
    subsets = subsets or ["train"]
    datasets = load_dataset(dataset_name)
    datasets = datasets.rename_column(key_column, NEW_KEY_COLUMN)
    datasets = datasets.select_columns([NEW_KEY_COLUMN])
    dataset_list = []
    for subset in subsets:
        dataset = datasets[subset]
        dataset = dataset.add_column("source", [dataset_name] * len(dataset))
        dataset = dataset.add_column("split", [subset] * len(dataset))
        dataset_list.append(dataset)
    dataset = concatenate_datasets(dataset_list)
    return dataset


REWRITE_PROMPTS = [
    "Rewrite in the style of a sea shanty.",
    "Improve the text.",
    "Rewrite this essay but do it using the writing style of Dr. Seuss.",
    "Rewrite this essay but do it using the writing style of William Shakespeare.",
    "Rewrite this essay but do it using the writing style of Tupac Shakur.",]

REWRITE_TEMPLATE = "{rewrite_prompt} {original_text}"