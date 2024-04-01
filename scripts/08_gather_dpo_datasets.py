import os

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from dotenv import load_dotenv
from loguru import logger

import wandb

load_dotenv()
SPLITS = ["train", "validation", "test"]

INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "../input")
INPUT_DATASET_TYPE = "generated_and_scored_texts"
OUTPUT_DATASET_NAME = "gathered_dpo_texts"

KEEP_COLUMNS = ["id", "original_text", "rewritten_text", "rewrite_prompt", "source", "chosen", "rejected", "version"]


def convert_dataset_to_dpo(dataset: Dataset) -> Dataset:
    df = dataset.to_pandas()
    df["id"] = df.index
    df["best"] = df["cosine"].apply(lambda x: np.array(x).argmax())
    df["chosen"] = df.apply(lambda x: x["predicted"][x["best"]], axis=1)
    df = df.explode("predicted")
    df["count"] = df.groupby((df["id"] != df["id"].shift(1)).cumsum()).cumcount()
    df = df.query("count != best").reset_index(drop=True)
    df = df.rename(columns={"predicted": "rejected"})
    return Dataset.from_pandas(df)


def main() -> None:
    run = wandb.init(job_type="gather_dpo_datasets")
    wandb_api = wandb.Api()
    artifact_collection = wandb_api.artifact_collections("llm-prompt-recovery", INPUT_DATASET_TYPE)
    dataset_dict = {k: [] for k in SPLITS}
    for elem in artifact_collection:
        logger.info(f"Downloading artifact {elem.name}")
        artifact = run.use_artifact(f"{elem.name}:latest", type=INPUT_DATASET_TYPE)
        datadir = artifact.download(f"./artifacts/{INPUT_DATASET_TYPE}/{elem.name}")
        loaded_dataset = load_from_disk(datadir)
        for key, dataset in loaded_dataset.items():
            dataset = convert_dataset_to_dpo(dataset)
            dataset = dataset.add_column("version", [elem.name] * len(dataset))
            dataset = dataset.select_columns(KEEP_COLUMNS)
            dataset_dict[key].append(dataset)
    dataset_dict = {k: concatenate_datasets(v) for k, v in dataset_dict.items()}
    dataset_dict = DatasetDict(dataset_dict)

    for key, dataset in dataset_dict.items():
        logger.info(f"Dataset {key} has {len(dataset)} samples")

    dataset_dict.save_to_disk(f"{INPUT_DATA_DIR}/{OUTPUT_DATASET_NAME}")
    artifact = wandb.Artifact(OUTPUT_DATASET_NAME, type="dataset")
    artifact.add_dir(f"{INPUT_DATA_DIR}/{OUTPUT_DATASET_NAME}")
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()
