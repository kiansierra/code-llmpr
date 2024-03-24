import argparse
import os

from datasets import load_from_disk
from dotenv import load_dotenv

import wandb
from llm_prompt import EnglishLabeler, ResponsePollutionLabeler

load_dotenv()
SPLITS = ['train', 'validation', 'test']

INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "../input")
INPUT_DATASET_NAME = "rewritten_texts"
OUTPUT_DATASET_NAME = "labeled_rewritten_texts"

def parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--version", type=int, default=1)
    argparser.add_argument("--downloaded", action="store_true", default=False)
    return argparser.parse_args()

def main(args) -> None:
    run = wandb.init(job_type='label_rewriten_texts', config=vars(args))
    prefix = f'v-{args.version}'
    if args.downloaded:
        prefix = 'v-downloaded'
    artifact = run.use_artifact(f"{INPUT_DATASET_NAME}:latest")
    datadir = artifact.download(f'./artifacts/{INPUT_DATASET_NAME}', path_prefix=prefix)
    datasets = load_from_disk(f"{datadir}/{prefix}")
    with EnglishLabeler() as labeler:
        datasets = datasets.map(labeler,
                                batched=True,
                                batch_size=64,
                                desc="Labeling English texts")
    with ResponsePollutionLabeler() as labeler:
        datasets = datasets.map(labeler,
                                batched=True,
                                batch_size=64,
                                desc="Labeling Polluted responses")
    datasets.save_to_disk(f"{INPUT_DATA_DIR}/{OUTPUT_DATASET_NAME}/{prefix}")
    artifact = wandb.Artifact(OUTPUT_DATASET_NAME, type="dataset")
    artifact.add_dir(f"{INPUT_DATA_DIR}/{OUTPUT_DATASET_NAME}")
    run.log_artifact(artifact)
    run.finish()
    
if __name__ == "__main__":
    args = parser()
    main(args)