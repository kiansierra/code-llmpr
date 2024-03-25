import argparse
import os

import numpy as np
from datasets import DatasetDict, load_from_disk
from dotenv import load_dotenv
import hydra
import wandb
from llm_prompt import REWRITE_TEMPLATES, APIGenerator, GemmaGenerator, EnglishLabeler
from omegaconf import OmegaConf
load_dotenv()

VARIANT = "7b-it-quant"
WEIGHTS_DIR = "../checkpoints/7b-it-quant"

INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "../input")
INPUT_DATASET_NAME = "templates"
OUTPUT_DATASET_TYPE = "rewritten_texts"

INSTRUCTION_PROMPT = "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


@hydra.main(config_path="../src/llm_prompt/configs/scripts", config_name="03_rewritte.yaml", version_base=None)
def main(args):
    if args.split not in ["train", "validation", "test", "all"]:
        raise ValueError(f"Unkown split {args.split}. Please use one of the following: train, validation, test, all.")
    if args.split == "all":
        splits = ["train", "validation", "test"]
    else:
        splits = [args.split]
    save_path = f"{INPUT_DATA_DIR}/{OUTPUT_DATASET_TYPE}/v-{args.version}"
    if os.path.exists(save_path):
        raise ValueError(f"Path {save_path} already exists. Please remove it before running this script.")

    if args.seed is not None:
        args.seed = args.version
    dataset_name = f"v-{args.version}"
    resolved_config = OmegaConf.to_container(args, resolve=True)
    run = wandb.init(job_type="rewrite_text", config=resolved_config)
    artifact = run.use_artifact(f"{INPUT_DATASET_NAME}:latest")
    datadir = artifact.download(f"./artifacts/{INPUT_DATASET_NAME}")
    dd = load_from_disk(datadir)
    # Shuffle the dataset and sample for each split
    dd = dd.shuffle(args.seed)
    dataset_dict = {}
    for key, dataset in dd.items():
        if key not in splits:
            continue
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
        dataset_dict[key] = dataset
    dd = DatasetDict(dataset_dict)
    ## Add probability of english and filter out texts with low probability
    with EnglishLabeler() as labeler:
        dd = dd.map(labeler, batched=True, batch_size=64, desc="Labeling English texts")
    dd = dd.filter(lambda x: x["en"] > args.prob_en, desc=f"Filtering texts with english prob above {args.prob_en}")
    ## Create the Input for generation
    dd = dd.map(
        lambda x: {"input": INSTRUCTION_PROMPT.format(prompt=np.random.choice(REWRITE_TEMPLATES).format(**x))},
        desc="Rewriting prompts",
        num_proc=4,
    )
    if args.online:
        generator = APIGenerator("google/gemma-7b-it")
    else:
        generator = GemmaGenerator(VARIANT, WEIGHTS_DIR, {"output_len": args.output_len, "top_k": args.top_k})
    generator.setup()
    dd = dd.map(
        generator.generate_batch,
        batched=True,
        batch_size=args.batch_size,
        input_columns=["input"],
        desc="Generating rewritten text",
    )
    dd = dd.filter(lambda x: x["rewritten_text"] != "EMPTY", desc="Filtering out empty generated texts")
    save_path = f"{INPUT_DATA_DIR}/{OUTPUT_DATASET_TYPE}/{dataset_name}"
    dd.save_to_disk(save_path)
    artifact = wandb.Artifact(f"{dataset_name}-{OUTPUT_DATASET_TYPE}", type=OUTPUT_DATASET_TYPE)
    artifact.add_dir(save_path)
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
