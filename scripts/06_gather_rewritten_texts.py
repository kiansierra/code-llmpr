import argparse
import os
from pathlib import Path

import wandb
from datasets import DatasetDict, concatenate_datasets, load_from_disk
from dotenv import load_dotenv
from loguru import logger

load_dotenv()
SPLITS = ['train', 'validation', 'test']

INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "../input")
INPUT_DATASET_NAME = "labeled_rewritten_texts"
OUTPUT_DATASET_NAME = "gathered_rewritten_texts"

KEEP_COLUMNS = ['original_text', 'rewritten_text', 'rewrite_prompt', 'source', 'yes', 'en']

def parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--yes", type=float, default=0.7)
    argparser.add_argument("--en", type=float, default=0.8)
    return argparser.parse_args()

def main(args) -> None:
    
    run = wandb.init(job_type='gather_rewriten_texts', config=vars(args))
    artifact = run.use_artifact(f"{INPUT_DATASET_NAME}:latest")
    datadir = artifact.download(f'./artifacts/{INPUT_DATASET_NAME}')
    all_datasets = list(Path(datadir).glob("v-*"))
    dataset_dict = {k:[] for k in SPLITS}
    for path in all_datasets:
        loaded_dataset = load_from_disk(path)
        for key, dataset in loaded_dataset.items():
            dataset = dataset.select_columns(KEEP_COLUMNS)
            dataset.add_column('version', [path.name]*len(dataset))
            dataset_dict[key].append(dataset)
    dataset_dict = {k: concatenate_datasets(v) for k,v in dataset_dict.items()}
    dataset_dict = DatasetDict(dataset_dict)
    dataset_dict = dataset_dict.filter(lambda x: x['en'] > args.en,
                        desc=f"Filtering texts with english prob above {args.en}")
    dataset_dict =  dataset_dict.filter(lambda x: x['yes'] < args.yes,
                        desc=f"Filtering texts with probability of containing promptin instructions below {args.yes}")
    
    for key, dataset in dataset_dict.items():
        logger.info(f"Dataset {key} has {len(dataset)} samples")
    
    dataset_dict.save_to_disk(f"{INPUT_DATA_DIR}/{OUTPUT_DATASET_NAME}")
    artifact = wandb.Artifact(OUTPUT_DATASET_NAME, type="dataset")
    artifact.add_dir(f"{INPUT_DATA_DIR}/{OUTPUT_DATASET_NAME}")
    run.log_artifact(artifact)
    run.finish()
    
if __name__ == "__main__":
    args = parser()
    main(args)