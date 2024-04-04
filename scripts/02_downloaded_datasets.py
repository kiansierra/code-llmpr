import argparse
import os

import pandas as pd
from datasets import Dataset, DatasetDict
import numpy as np
import wandb
from llm_prompt import get_configs

INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "../input")
OUTPUT_DATASET_TYPE = "rewritten_texts"

KEEP_COLUMNS = ["original_text", "rewritten_text", "rewrite_prompt", "source"]

def parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument("--split", type=float, default=0.1)
    return argparser.parse_args()


def gather_downloaded_datasets(seed:int) -> pd.DataFrame:
    dataset_configs = get_configs("dataset/downloaded")
    all_dfs = []
    for name, config in dataset_configs.items():
        for file in config.files:
            df = pd.read_csv(f"../input/{config.folder}/{file}")
            df["source"] = name
            df["file"] = file
            all_dfs.append(df)
    df = pd.concat(all_dfs, ignore_index=True)
    df = df[KEEP_COLUMNS]
    df = df.dropna().reset_index(drop=True)
    np.random.seed(seed)
    df["split"] = np.random.choice(["train", "validation", "test"], size=len(df), p=[0.8, 0.1, 0.1])
    return df


def main():
    args = parser()
    version = "downloaded"
    df = gather_downloaded_datasets(args.seed)
    dataset = Dataset.from_pandas(df)
    dataset_dict = DatasetDict({key: dataset.filter(lambda x: x["split"] == key).select_columns(KEEP_COLUMNS) 
                                for key in ["train", "validation", "test"]})
    run = wandb.init(job_type="downloaded_texts", config=vars(args))
    dataset_name = f"v-{version}"
    dataset_dict.save_to_disk(f"{INPUT_DATA_DIR}/{OUTPUT_DATASET_TYPE}/{dataset_name}")
    artifact = wandb.Artifact(f"{dataset_name}-{OUTPUT_DATASET_TYPE}", type=OUTPUT_DATASET_TYPE)
    artifact.add_dir(f"{INPUT_DATA_DIR}/{OUTPUT_DATASET_TYPE}/{dataset_name}")
    run.log_artifact(artifact)
    prompt_df = df[['source', 'rewrite_prompt']].drop_duplicates().reset_index(drop=True)
    prompt_df.to_parquet(f"{INPUT_DATA_DIR}/prompts.parquet")
    prompt_artifact = wandb.Artifact("prompts", type="dataset")
    prompt_table = wandb.Table(dataframe=prompt_df)
    prompt_artifact.add_file(f"{INPUT_DATA_DIR}/prompts.parquet")
    prompt_artifact.add(prompt_table, "prompts")
    run.log_artifact(prompt_artifact)
    run.finish()


if __name__ == "__main__":
    main()
