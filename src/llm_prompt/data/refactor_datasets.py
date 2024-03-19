from typing import List, Optional

from datasets import Dataset, concatenate_datasets, load_dataset

NEW_KEY_COLUMN = "original_text"

__all__ = ["dataset_preprocess", "REWRITE_PROMPTS", "REWRITE_TEMPLATE"]

def dataset_preprocess(dataset_name: str, key_column: str ,subsets:Optional[List[str]] = None) -> Dataset:
    subsets = subsets or ["train"]
    datasets = load_dataset(dataset_name)
    datasets = datasets.rename_column(key_column, NEW_KEY_COLUMN)
    datasets = datasets.select_columns([NEW_KEY_COLUMN])
    datasets = datasets.map(lambda x: {"original_text": x[NEW_KEY_COLUMN]}, remove_columns=[NEW_KEY_COLUMN])
    dataset_list = []
    for subset in subsets:
        dataset = datasets[subset]
        dataset = dataset.add_column("source", [dataset_name] * len(dataset))
        dataset = dataset.add_column("split", [subset] * len(dataset))
        dataset = dataset.map(lambda x: {'original_length': len(x.split(" "))}, input_columns=NEW_KEY_COLUMN)
        dataset_list.append(dataset)
    dataset = concatenate_datasets(dataset_list)
    return dataset

AUTHORS = ["Dr. Seuss", "William Shakespeare", "Tupac Shakur",
           "J.K Rowling", "Stephen King", "JRR Tolkien", "Paulo Coelho"]

STYLES = ["sea shanty", "rap song", "poem", "haiku", "limerick", "sonnet", "ballad", "ode", "epic"]

AUTHOR_PROMPTS = [
    "Rewrite this essay but do it using the writing style of {author}.",
    "Transform this text as if it was written by {author}.",
    "Imagine {author} was to rewrite this text, what would it be like."
]

STYLE_PROMPTS = [
    "Rewrite this text in the style of a {style}.",
    "Transform this in to a {style}.",
    "How would you rewrite this text in the style of a {style}."
]

BASIC_REWRITE_PROMPTS = [
    "Improve the text.",
    "Make this text better.",
    "Rewrite this essay.",
    "Improve this essay.",
    "Make this essay better.",
    "Summarize this text.",
]

ALL_AUTHOR_PROMPTS = [prompt.format(author=author) for author in AUTHORS for prompt in AUTHOR_PROMPTS]
ALL_STYLE_PROMPTS = [prompt.format(style=style) for style in STYLES for prompt in STYLE_PROMPTS]

REWRITE_PROMPTS = BASIC_REWRITE_PROMPTS + ALL_AUTHOR_PROMPTS + ALL_STYLE_PROMPTS

REWRITE_TEMPLATE = "{rewrite_prompt} {original_text}"
