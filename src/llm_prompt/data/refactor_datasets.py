from typing import List, Optional
from omegaconf import OmegaConf, DictConfig
from datasets import Dataset, concatenate_datasets, load_dataset

NEW_KEY_COLUMN = "original_text"

__all__ = ["dataset_preprocess", "REWRITE_PROMPTS", "REWRITE_TEMPLATES"]

def dataset_preprocess(dataset_name: str,
                       key_column: str,
                       subset:Optional[str]=None,
                       splits:Optional[List[str]] = None,
                       data_files:Optional[str]= None) -> Dataset:
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
        dataset = dataset.map(lambda x: {'original_length': len(x.split(" "))}, input_columns=NEW_KEY_COLUMN)
        dataset_list.append(dataset)
    dataset = concatenate_datasets(dataset_list)
    return dataset

AUTHORS = ["Dr. Seuss", "William Shakespeare", "Tupac Shakur", "J.K Rowling", "Stephen King", "JRR Tolkien",
           "Paulo Coelho", "Taylor Swift", "Shakespeare", "Jane Austen", "Charles Dickens", "Mark Twain",]

STYLES = ["sea shanty", "rap song", "poem", "haiku", "limerick", "sonnet", "ballad", "ode", "epic"]

HISTORICAL_PERIODS = ["Victorian era", "Elizabethan era", "Renaissance", "Medieval", "Ancient Greek", "Ancient Roman",
                      "Middle Ages", "Enlightenment", "Baroque", "Romantic", "Modernist", "Postmodernist"]

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

PERIOD_PROMPTS = [
    "Rewrite the text as if it were written during the {period} historical period.",
    "Transform this text using language and cultural references appropriate to the {period} time period"
]

BASIC_REWRITE_PROMPTS = [
    "Improve the text.",
    "Make this text better.",
    "Rewrite this essay.",
    "Improve this essay.",
    "Make this essay better.",
    "Summarize this text.",
    "Rewrite the text using formal language suitable for an academic journal.",
    "Create a compelling plot that engages the reader.",
    "Rewrite the text in a humorous tone, using wit, sarcasm, and comedic elements.",
    "Use rhetorical devices and persuasive language techniques.",
    "Rewrite the text as if you're having a friendly conversation with a reader.",
    "Rewrite the text using poetic language, imagery, and metaphors.",
    "Focus on creating a rhythmic flow and evoking emotions through language.",
    "Recast the text using poetic language, metaphors, and imagery to create a rhythmic and evocative piece of writing.",
    "Rewrite the text as a travelogue, describing experiences, sights, and sounds in vivid detail to transport the reader to different locations."
]

ALL_AUTHOR_PROMPTS = [prompt.format(author=author) for author in AUTHORS for prompt in AUTHOR_PROMPTS]
ALL_STYLE_PROMPTS = [prompt.format(style=style) for style in STYLES for prompt in STYLE_PROMPTS]
ALL_PERIOD_PROMPTS = [prompt.format(period=period) for period in HISTORICAL_PERIODS for prompt in PERIOD_PROMPTS]

REWRITE_PROMPTS = BASIC_REWRITE_PROMPTS + ALL_AUTHOR_PROMPTS + ALL_STYLE_PROMPTS + ALL_PERIOD_PROMPTS

REWRITE_TEMPLATES = ["{rewrite_prompt} {original_text}",
    "Given the prompt: {rewrite_prompt}, rewrite the following text: {original_text}",
    "Given the prompt: {rewrite_prompt}, rewrite the following text, don't mention anything about the task at hand: {original_text}",]
