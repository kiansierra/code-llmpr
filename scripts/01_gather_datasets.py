import os

from datasets import DatasetDict
from dotenv import load_dotenv
from loguru import logger

import wandb
from llm_prompt import dataset_preprocess, get_configs

load_dotenv()

MAX_WORDS = 150
NUM_PROC = 8
DATASET_NAME = "raw_texts"
INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "../input")


def main() -> None:
    run = wandb.init(job_type="gather_datasets")
    dataset_configs = get_configs("dataset/generation")

    dataset_dict = {}
    for key, val in dataset_configs.items():
        dataset_dict[key] = dataset_preprocess(**val)
        logger.info(f"Dataset: {key} Original Length: {len(val)}")
    dataset_dict = DatasetDict(dataset_dict)
    dataset_dict = dataset_dict.filter(
        lambda x: x["original_length"] < MAX_WORDS,
        desc=f"Filtering texts with over {MAX_WORDS} words",
        num_proc=NUM_PROC,
    )
    for key, value in dataset_dict.items():
        logger.info(f"Dataset: {key} Cleaned up Length: {len(value)}")
    os.makedirs(INPUT_DATA_DIR, exist_ok=True)
    dataset_dict.save_to_disk(f"{INPUT_DATA_DIR}/{DATASET_NAME}")
    artifact = wandb.Artifact(DATASET_NAME, type="dataset")
    artifact.add_dir(f"{INPUT_DATA_DIR}/{DATASET_NAME}")
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()
