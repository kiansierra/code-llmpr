import argparse
import os

import numpy as np
from datasets import DatasetDict, load_from_disk

import wandb
from llm_prompt import REWRITE_TEMPLATES, GemmaGenerator

VARIANT = "7b-it-quant"
WEIGHTS_DIR = '../checkpoints/7b-it-quant'

NUM_PROMPTS_PER_TEXT = 4
INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "../input")
INPUT_DATASET_NAME = "templates"
OUTPUT_DATASET_NAME = "rewritten_texts"

INSTRUCTION_PROMPT = "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

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
    
    run = wandb.init(job_type='rewrite_text', config=vars(args))
    artifact = run.use_artifact(f"{INPUT_DATASET_NAME}:latest")
    datadir = artifact.download(f'./artifacts/{INPUT_DATASET_NAME}')
    dd = load_from_disk(datadir)
    dd = dd.map(lambda x: {'input': INSTRUCTION_PROMPT.format(prompt=np.random.choice(REWRITE_TEMPLATES).format(**x))},
                desc="Rewriting prompts",
                num_proc=4)
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
    dd.save_to_disk(f'{INPUT_DATA_DIR}/{OUTPUT_DATASET_NAME}/v-{args.version}')
    artifact = wandb.Artifact(OUTPUT_DATASET_NAME, type="dataset")
    artifact.add_dir(f"{INPUT_DATA_DIR}/{OUTPUT_DATASET_NAME}")
    run.log_artifact(artifact)
    run.finish()
    
if __name__ == "__main__":
    args = parser()
    main(args)