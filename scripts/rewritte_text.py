from datasets import DatasetDict, load_from_disk, concatenate_datasets
import shutil
from llm_prompt import REWRITE_TEMPLATE, GemmaGenerator
import argparse
import os
VARIANT = "7b-it-quant"
WEIGHTS_DIR = '../checkpoints/7b-it-quant'
NUM_SAMPLES = 2000

def parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--num_samples", type=int, default=2000)
    argparser.add_argument("--version", type=int, default=1)
    argparser.add_argument("--output_len", type=int, default=200)
    argparser.add_argument("--top_k", type=int, default=10)
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument("--batch_size", type=int, default=16)
    argparser.add_argument("--split", type=str, default="train")    
    return argparser.parse_args()
    

def main(args):
    if args.split not in ["train", "validation", "test", "all"]:
        raise ValueError(f"Unkown split {args.split}. Please use one of the following: train, validation, test, all.")
    if args.split == "all":
        splits = ["train", "validation", "test"]
    else:
        splits = [args.split]
    save_path = f'../input/rewritten_texts/v-{args.version}'
    if os.path.exists(save_path):
        raise ValueError(f"Path {save_path} already exists. Please remove it before running this script.")
    
    dd = load_from_disk('../input/templates')
    dd = dd.map(lambda x: {'input': REWRITE_TEMPLATE.format(**x)}, desc="Rewriting prompts", num_proc=4)
    generator = GemmaGenerator(VARIANT, WEIGHTS_DIR, {"output_len": args.output_len, "top_k":args.top_k})
    generator.setup()
    dd = dd.shuffle(args.seed)
    dataset_dict = {}
    for key, dataset in dd.items():
        if key not in splits:
            continue
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
        dataset = dataset.map(generator.generate_batch,
                              batched=True,
                              batch_size=args.batch_size,
                              input_columns=['input'],
                              desc=f"Generating rewritten text for {key}")
        dataset_dict[key] = dataset
    dd = DatasetDict(dataset_dict)
    dd.save_to_disk(f'../input/rewritten_texts/v-{args.version}')
    
if __name__ == "__main__":
    args = parser()
    main(args)