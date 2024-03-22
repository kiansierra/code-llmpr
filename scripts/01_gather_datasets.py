import os

from datasets import DatasetDict
import wandb
from llm_prompt import dataset_preprocess, get_configs
from dotenv import load_dotenv

load_dotenv()

MAX_WORDS = 150
NUM_PROC = 8
DATASET_NAME="raw_texts"
INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "../input")

def main() -> None:
    run = wandb.init(job_type='gather_datasets')
    dataset_configs = get_configs('dataset')
    dataset_dict = DatasetDict({key: dataset_preprocess(**val) for key, val in dataset_configs.items()})
    dataset_dict = dataset_dict.filter(lambda x: x['original_length'] < MAX_WORDS,
                                       desc=f"Filtering texts with over {MAX_WORDS} words",
                                       num_proc=NUM_PROC)
    os.makedirs(INPUT_DATA_DIR, exist_ok=True)
    dataset_dict.save_to_disk(f"{INPUT_DATA_DIR}/{DATASET_NAME}")
    artifact = wandb.Artifact(DATASET_NAME, type="dataset")
    artifact.add_dir(f"{INPUT_DATA_DIR}/{DATASET_NAME}")
    run.log_artifact(artifact)
    run.finish()
    
if __name__ == "__main__":
    main()