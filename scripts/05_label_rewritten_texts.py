import argparse
import os

from datasets import load_from_disk
from dotenv import load_dotenv

import wandb
from llm_prompt import ResponsePollutionLabeler

load_dotenv()
SPLITS = ["train", "validation", "test"]

INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "../input")
INPUT_DATASET_TYPE = "rewritten_texts"
OUTPUT_DATASET_TYPE = "labeled_rewritten_texts"


def parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--version", type=int, default=1)
    argparser.add_argument("--downloaded", action="store_true", default=False)
    return argparser.parse_args()


def main() -> None:
    args = parser()
    run = wandb.init(job_type="label_rewriten_texts", config=vars(args))
    prefix = f"v-{args.version}"
    if args.downloaded:
        prefix = "v-downloaded"
    artifact = run.use_artifact(f"{prefix}-{INPUT_DATASET_TYPE}:latest", type=INPUT_DATASET_TYPE)
    datadir = artifact.download(f"./artifacts/{INPUT_DATASET_TYPE}/{prefix}")
    datasets = load_from_disk(datadir)
    with ResponsePollutionLabeler() as labeler:
        datasets = datasets.map(labeler, batched=True, batch_size=64, desc="Labeling Polluted responses")
    save_dir = f"{INPUT_DATA_DIR}/{OUTPUT_DATASET_TYPE}/{prefix}"
    datasets.save_to_disk(save_dir)
    artifact = wandb.Artifact(f"{prefix}-{OUTPUT_DATASET_TYPE}", type=OUTPUT_DATASET_TYPE)
    artifact.add_dir(save_dir)
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()
