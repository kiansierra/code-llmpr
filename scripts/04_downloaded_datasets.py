import argparse
import os

import pandas as pd
from datasets import Dataset

import wandb
from llm_prompt import get_configs

INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "../input")
OUTPUT_DATASET_TYPE = "rewritten_texts"

def parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument("--split", type=float, default=0.1)
    return argparser.parse_args()

def gather_downloaded_datasets():
    keep_columns = ['original_text', 'rewritten_text', 'rewrite_prompt', 'source']
    dataset_configs = get_configs('dataset/downloaded')
    all_dfs = []
    for name, config in dataset_configs.items():
        for file in config.files:
            df = pd.read_csv(f"../input/{config.folder}/{file}")
            df['source'] = name
            df['file'] = file
            all_dfs.append(df)
    df = pd.concat(all_dfs, ignore_index=True)
    df = df[keep_columns]
    df = df.dropna().reset_index(drop=True)
    return df

    

def main(args):
    version = 'downloaded'
    df = gather_downloaded_datasets()
    dataset = Dataset.from_pandas(df)
    dd = dataset.train_test_split(test_size=0.1, seed=args.seed)
    new_dd = dd['train'].train_test_split(test_size=0.1, seed=args.seed)
    dd['validation'] = new_dd['test']
    dd['train'] = new_dd['train']
    run = wandb.init(job_type='downloaded_texts', config=vars(args))
    dataset_name = f"v-{version}"
    dd.save_to_disk(f'{INPUT_DATA_DIR}/{OUTPUT_DATASET_TYPE}/{dataset_name}')
    artifact = wandb.Artifact(f"{dataset_name}-{OUTPUT_DATASET_TYPE}", type=OUTPUT_DATASET_TYPE)
    artifact.add_dir(f"{INPUT_DATA_DIR}/{OUTPUT_DATASET_TYPE}/{dataset_name}")
    run.log_artifact(artifact)
    run.finish()
    
if __name__ == "__main__":
    args = parser()
    main(args)