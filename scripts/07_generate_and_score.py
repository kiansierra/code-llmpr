import argparse
import os
from itertools import zip_longest
from pathlib import Path

import torch
from datasets import load_from_disk
from dotenv import load_dotenv
from omegaconf import OmegaConf
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

import wandb
from llm_prompt import CosineScorer

load_dotenv()
SPLITS = ['train', 'validation', 'test']

INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "../input")
INPUT_DATASET_TYPE = "labeled_rewritten_texts"
INPUT_MODEL_NAME = "Llama-2-7b-hf"

OUTPUT_DATASET_TYPE = "generated_and_scored_texts"

KEEP_COLUMNS = ['original_text', 'rewritten_text', 'rewrite_prompt', 'source']

def parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--version", type=int, required=True)
    argparser.add_argument("--output_len", type=int, default=200)
    argparser.add_argument("--top_k", type=int, default=10)
    argparser.add_argument("--seed", type=int, default=None)
    argparser.add_argument("--batch_size", type=int, default=4)
    argparser.add_argument("--downloaded", action="store_true", default=False)
    return argparser.parse_args()

RESPONSE_TEMPLATE = "### Prompt Used: "

def grouper(iterable, n, fillvalue=None):
    items = [iter(iterable)] * n
    return list(map(list, zip_longest(*items, fillvalue=fillvalue)))

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['original_text'])):
        text = f"""### Original Text: {example['original_text'][i]} ### Rewriten Text: {example['rewritten_text'][i]} {RESPONSE_TEMPLATE} """
        output_texts.append(text)
    return output_texts

def batch_generator(model, tokenizer, **generation_kwargs):
    num_return_sequences = generation_kwargs.get('num_return_sequences', 1)
    generation_kwargs_add = {k:[v]*num_return_sequences for k,v in generation_kwargs.items() if k not in ['num_return_sequences']}
    generation_kwargs['pad_token_id'] = tokenizer.eos_token_id
    def generate(batch):
        inputs = tokenizer(batch["input"], return_tensors="pt", padding=True).to(model.device)
        outputs = model.generate(**inputs,**generation_kwargs)
        end_of_inputs = inputs.attention_mask.shape[1]
        predicted = tokenizer.batch_decode(outputs[:, end_of_inputs:], skip_special_tokens=True)
        predicted = grouper(predicted, num_return_sequences)
        return {"predicted": predicted, **generation_kwargs_add}
    return generate

def main(args) -> None:
    
    run = wandb.init(job_type='generate_and_score', config=vars(args))
    datadir = f'./artifacts/{INPUT_MODEL_NAME}'
    artifact = run.use_artifact(f"{INPUT_MODEL_NAME}:latest")
    model_datadir = artifact.download(datadir)
    config = OmegaConf.load(f"{model_datadir}/config.yaml")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                                bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(**config.model, quantization_config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    last_checkpoint = max([int(elem.name.replace("checkpoint-", "")) for elem in Path(datadir).glob("checkpoint-*")])
    model.load_adapter(F"{datadir}/checkpoint-{last_checkpoint}")
    dataset_name = f'v-{args.version}'
    if args.downloaded:
        dataset_name = 'v-downloaded'
    artifact = run.use_artifact(f"{dataset_name}-{INPUT_DATASET_TYPE}:latest", type=INPUT_DATASET_TYPE)
    datadir = artifact.download(f'./artifacts/{INPUT_DATASET_TYPE}/{dataset_name}')
    dataset = load_from_disk(datadir)
    dataset = dataset.map(lambda examples: {"input":formatting_prompts_func(examples)}, batched=True)
    generation_kwargs = {'num_return_sequences': 4, 'max_new_tokens': 20, 'temperature': 0.8, 'top_k': 50, 'top_p': 0.9}
    dataset = dataset.map(batch_generator(model, tokenizer, **generation_kwargs),
                          batched=True,
                          batch_size=args.batch_size,
                          desc="Generating texts")
    del model
    torch.cuda.empty_cache()
    with CosineScorer() as scorer:
        dataset = dataset.map(scorer, batched=True, batch_size=64, desc="Scoring generated texts")
    
    dataset.save_to_disk(f"{INPUT_DATA_DIR}/{OUTPUT_DATASET_TYPE}/{dataset_name}")
    artifact = wandb.Artifact(OUTPUT_DATASET_TYPE, type=OUTPUT_DATASET_TYPE)
    artifact.add_dir(f"{INPUT_DATA_DIR}/{OUTPUT_DATASET_TYPE}/{dataset_name}")
    run.log_artifact(artifact)
    run.finish()
    
if __name__ == "__main__":
    args = parser()
    main(args)