import os
from llm_prompt import dataset_preprocess, get_configs
from datasets import DatasetDict


def main() -> None:
    dataset_configs = get_configs('dataset')
    dataset_dict = DatasetDict({key: dataset_preprocess(**val) for key, val in dataset_configs.items()})
    os.makedirs("../inputs", exist_ok=True)
    dataset_dict.save_to_disk("../input/raw")
    
if __name__ == "__main__":
    main()