
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
import numpy as np
from llm_prompt import REWRITE_PROMPTS

SPLITS = ['train', 'validation', 'test']

MAX_WORDS = 150
NUM_PROMPTS_PER_TEXT = 4

def main() -> None:
    raw_dataset_dict = load_from_disk("../input/raw")
    dataset_templates = []
    for dataset in raw_dataset_dict.values():
        dataset= dataset.filter(lambda x: x['original_length'] < MAX_WORDS)
        df = dataset.to_pandas()
        df['rewrite_prompt']  = df.apply(lambda x: np.random.choice(REWRITE_PROMPTS, NUM_PROMPTS_PER_TEXT), axis=1)
        df = df.explode('rewrite_prompt')
        dataset_templates.append(Dataset.from_pandas(df))
    dataset = concatenate_datasets(dataset_templates)
    dataset_dict = {key: dataset.filter(lambda x: x['split'] == key) for key in SPLITS}
    DatasetDict(dataset_dict).save_to_disk("../input/templates")

    
    
if __name__ == "__main__":
    main()