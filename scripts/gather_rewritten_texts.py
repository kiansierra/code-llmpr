from pathlib import Path
from datasets import load_from_disk, concatenate_datasets, DatasetDict

SPLITS = ['train', 'validation', 'test']

def main():
    all_datasets = list(Path("../input/rewritten_texts").glob("v-*"))
    dataset_dict = {k:[] for k in SPLITS}
    for path in all_datasets:
        loaded_dataset = load_from_disk(path)
        for key, dataset in loaded_dataset.items():
            dataset_dict[key].append(dataset)
    dataset_dict = {k: concatenate_datasets(v) for k,v in dataset_dict.items()}
    DatasetDict(dataset_dict).save_to_disk("../input/rewritten")
    
if __name__ == "__main__":
    main()