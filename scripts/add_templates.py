import os
from llm_prompt import REWRITE_TEMPLATES
from datasets import DatasetDict, Dataset, load_from_disk
import pandas as pd

def main() -> None:
    dataset_dict = load_from_disk("../inputs/raw")
    templates_df = pd.DataFrame({
        'rewrite_prompt': REWRITE_TEMPLATES,
        'key': [1]*len(REWRITE_TEMPLATES)
    })
    dataset_dict_templates = {}
    for key, dataset in dataset_dict.items():
        df = dataset.to_pandas()
        df['key'] = 1
        merged_df = df.merge(templates_df, on='key').drop(columns='key')
        dataset_dict_templates[key] =  Dataset.from_pandas(merged_df)
        
    DatasetDict(dataset_dict_templates).save_to_disk("../inputs/templates")

    
    
if __name__ == "__main__":
    main()