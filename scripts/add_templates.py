
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

from llm_prompt import REWRITE_PROMPTS

SPLITS = ['train', 'validation', 'test']

def main() -> None:
    raw_dataset_dict = load_from_disk("../input/raw")
    templates_df = pd.DataFrame({
        'rewrite_prompt': REWRITE_PROMPTS,
        'key': [1]*len(REWRITE_PROMPTS)
    })
    dataset_templates = []
    for dataset in raw_dataset_dict.values():
        df = dataset.to_pandas()
        df['key'] = 1
        merged_df = df.merge(templates_df, on='key').drop(columns='key')
        dataset_templates.append(Dataset.from_pandas(merged_df))
    dataset = concatenate_datasets(dataset_templates)
    dataset_dict = {key: dataset.filter(lambda x: x['split'] == key) for key in SPLITS}
    DatasetDict(dataset_dict).save_to_disk("../input/templates")

    
    
if __name__ == "__main__":
    main()