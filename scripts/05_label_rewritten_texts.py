import argparse
import os

import torch
from datasets import load_from_disk
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import wandb

load_dotenv()
SPLITS = ['train', 'validation', 'test']

INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "../input")
INPUT_DATASET_NAME = "rewritten_texts"
OUTPUT_DATASET_NAME = "labeled_rewritten_texts"

def parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--version", type=int, default=1)
    return argparser.parse_args()

def label_batch(model, tokenizer):
    eng_id = model.config.label2id['en']
    
    @torch.no_grad()
    def _label_batch(batch):
        inputs = tokenizer(batch['original_text'], padding=True, truncation=True, return_tensors="pt").to(model.device)
        logits = model(**inputs).logits
        return {'en': logits[:, eng_id]}
    
    return _label_batch

def main(args) -> None:
    run = wandb.init(job_type='gather_rewriten_texts', config=vars(args))
    prefix = f'v-{args.version}'
    artifact = run.use_artifact(f"{INPUT_DATASET_NAME}:latest")
    datadir = artifact.download(f'./artifacts/{INPUT_DATASET_NAME}/{prefix}', path_prefix=prefix)
    datasets = load_from_disk(datadir)
    model_ckpt = "papluca/xlm-roberta-base-language-detection"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt).to("cuda")
    datasets = datasets.map(label_batch(model, tokenizer), batched=True, batch_size=16)
    datasets.save_to_disk(f"{INPUT_DATA_DIR}/{OUTPUT_DATASET_NAME}/{prefix}")
    artifact = wandb.Artifact(OUTPUT_DATASET_NAME, type="dataset")
    artifact.add_dir(f"{INPUT_DATA_DIR}/{OUTPUT_DATASET_NAME}")
    run.log_artifact(artifact)
    run.finish()
    
if __name__ == "__main__":
    args = parser()
    main(args)