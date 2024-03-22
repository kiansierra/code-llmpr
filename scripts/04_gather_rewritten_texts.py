from pathlib import Path
from datasets import load_from_disk, concatenate_datasets, DatasetDict
import os
import wandb 
from dotenv import load_dotenv

load_dotenv()
SPLITS = ['train', 'validation', 'test']

INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "../input")
INPUT_DATASET_NAME = "rewritten_texts"
OUTPUT_DATASET_NAME = "gathered_rewritten_texts"

def main() -> None:
    run = wandb.init(job_type='gather_rewriten_texts')
    artifact = run.use_artifact(f"{INPUT_DATASET_NAME}:latest")
    datadir = artifact.download(f'./artifacts/{INPUT_DATASET_NAME}')
    all_datasets = list(Path(datadir).glob("v-*"))
    dataset_dict = {k:[] for k in SPLITS}
    for path in all_datasets:
        loaded_dataset = load_from_disk(path)
        for key, dataset in loaded_dataset.items():
            dataset_dict[key].append(dataset)
    dataset_dict = {k: concatenate_datasets(v) for k,v in dataset_dict.items()}
    DatasetDict(dataset_dict).save_to_disk(f"{INPUT_DATA_DIR}/{OUTPUT_DATASET_NAME}")
    artifact = wandb.Artifact(OUTPUT_DATASET_NAME, type="dataset")
    artifact.add_dir(f"{INPUT_DATA_DIR}/{OUTPUT_DATASET_NAME}")
    run.log_artifact(artifact)
    run.finish()
    
if __name__ == "__main__":
    main()