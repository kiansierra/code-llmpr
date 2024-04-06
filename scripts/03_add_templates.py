import os

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from loguru import logger

import wandb

SPLITS = ["train", "validation", "test"]

NUM_PROMPTS_PER_TEXT = 4
INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "../input")
INPUT_TEXTS_DATASET_NAME = "raw_texts"
INPUT_PROMPTS_DATASET_NAME = "prompts"

OUTPUT_DATASET_NAME = "templates"


def main() -> None:
    run = wandb.init(job_type="add_templates")
    artifact = run.use_artifact(f"{INPUT_TEXTS_DATASET_NAME}:latest")
    datadir = artifact.download(f"./artifacts/{INPUT_TEXTS_DATASET_NAME}")
    raw_dataset_dict = load_from_disk(datadir)
    artifact = run.use_artifact(f"{INPUT_PROMPTS_DATASET_NAME}:latest")
    datadir = artifact.download(f"./artifacts/{INPUT_PROMPTS_DATASET_NAME}")
    prompts_df = pd.read_parquet(f"{datadir}/prompts.parquet")
    dataset_templates = []
    all_rewrite_prompts = prompts_df["rewrite_prompt"].tolist()
    probs = 1/prompts_df.groupby(['source'])['rewrite_prompt'].transform('count')
    probs /= probs.sum()
    prompts_df = prompts_df.rename(columns={"source": "prompt_source", "cluster": "prompt_cluster"})
    logger.info(f"Number of rewrite prompts: {len(all_rewrite_prompts)}")
    for name, dataset in raw_dataset_dict.items():
        logger.info(f"Adding prompts to {name}")
        selected_prompts = np.random.choice(all_rewrite_prompts, (len(dataset), NUM_PROMPTS_PER_TEXT), p=probs.values)
        dataset = dataset.add_column("rewrite_prompt", selected_prompts.tolist())
        df = dataset.to_pandas()
        df = df.explode("rewrite_prompt")
        logger.info(f"Merging prompts metadata to {name}")
        df = df.merge(prompts_df[["rewrite_prompt", "prompt_source", "prompt_cluster"]], on="rewrite_prompt")
        logger.info(f"Number of examples in {name}: {len(df)}")
        dataset_templates.append(Dataset.from_pandas(df))
    dataset = concatenate_datasets(dataset_templates)
    dataset_dict = {key: dataset.filter(lambda x: x["split"] == key, desc=f"Filtering {key}") for key in SPLITS}
    for key, dataset in dataset_dict.items():
        logger.info(f"Number of examples in {key}: {len(dataset)}")
    DatasetDict(dataset_dict).save_to_disk(f"{INPUT_DATA_DIR}/{OUTPUT_DATASET_NAME}")
    artifact = wandb.Artifact(OUTPUT_DATASET_NAME, type="dataset")
    artifact.add_dir(f"{INPUT_DATA_DIR}/{OUTPUT_DATASET_NAME}")
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()
