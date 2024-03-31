import os
from itertools import zip_longest
from pathlib import Path

import hydra
import numpy as np
import torch
from datasets import load_from_disk
from dotenv import load_dotenv
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizer

import wandb
from llm_prompt import FORMATTERS_MAPPING, CosineScorer

load_dotenv()

INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "../input")
INPUT_DATASET_TYPE = "labeled_rewritten_texts"
INPUT_MODEL_TYPE = "model-sft"

OUTPUT_DATASET_TYPE = "generated_and_scored_texts"


def grouper(iterable, n, fillvalue=None):
    items = [iter(iterable)] * n
    return list(map(list, zip_longest(*items, fillvalue=fillvalue)))


def batch_generator(model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizer, **generation_kwargs):
    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    generation_kwargs["pad_token_id"] = tokenizer.eos_token_id

    def generate(batch):
        inputs = tokenizer(batch["input"], return_tensors="pt", padding=True).to(model.device)
        outputs = model.generate(**inputs, **generation_kwargs)
        batch_size = inputs.attention_mask.shape[0]
        end_of_inputs = inputs.attention_mask.shape[1]
        predicted = tokenizer.batch_decode(outputs[:, end_of_inputs:], skip_special_tokens=True)
        predicted = grouper(predicted, num_return_sequences)
        generation_kwargs_add = {
            k: [v] * batch_size
            for k, v in generation_kwargs.items()
            if k not in ["num_return_sequences", "pad_token_id"]
        }
        return {"predicted": predicted, **generation_kwargs_add}

    return generate


@hydra.main(config_path="../src/llm_prompt/configs/scripts", config_name="07_generate.yaml", version_base=None)
def main(args) -> None:
    solved_config = OmegaConf.to_container(args, resolve=True)
    run = wandb.init(job_type="generate_and_score", config=solved_config)
    input_model_name = args.input_model_name
    datadir = f"./artifacts/{input_model_name}"
    artifact = run.use_artifact(f"{input_model_name}-sft:latest", INPUT_MODEL_TYPE)
    model_datadir = artifact.download(datadir)
    config = OmegaConf.load(f"{model_datadir}/config.yaml")
    quantization_config = BitsAndBytesConfig(**config.quantization)
    model = AutoModelForCausalLM.from_pretrained(
        **config.model, device_map="auto", quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    formatter = FORMATTERS_MAPPING[config.formatter](tokenizer)

    last_checkpoint = max([int(elem.name.replace("checkpoint-", "")) for elem in Path(datadir).glob("checkpoint-*")])
    model.load_adapter(f"{datadir}/checkpoint-{last_checkpoint}")
    dataset_name = f"v-{args.version}"
    if args.downloaded:
        dataset_name = "v-downloaded"
    artifact = run.use_artifact(f"{dataset_name}-{INPUT_DATASET_TYPE}:latest", type=INPUT_DATASET_TYPE)
    datadir = artifact.download(f"./artifacts/{INPUT_DATASET_TYPE}/{dataset_name}")
    dataset = load_from_disk(datadir)
    dataset = dataset.select_columns(["rewritten_text", "original_text", "rewrite_prompt", "source"])
    dataset = dataset.map(
        lambda x, y: {"input": formatter.format_row(x, y)},
        num_proc=4,
        desc="Formatting rows for generation",
        input_columns=["rewritten_text", "original_text"],
    )
    dataset = dataset.map(
        batch_generator(model, tokenizer, **args.generation_kwargs),
        batched=True,
        batch_size=args.batch_size,
        desc="Generating texts",
    )
    del model
    torch.cuda.empty_cache()
    with CosineScorer() as scorer:
        dataset = dataset.map(scorer, batched=True, batch_size=64, desc="Scoring generated texts")

    scores = {}
    for key, data in dataset.items():
        cosine = np.array(data["cosine"])
        scores[f"{key}/cosine_avg"] = cosine.mean()
        scores[f"{key}/cosine_std"] = cosine.std(1).mean()

    save_dir = f"{INPUT_DATA_DIR}/{OUTPUT_DATASET_TYPE}/{dataset_name}"
    dataset.save_to_disk(save_dir)
    artifact = wandb.Artifact(f"{dataset_name}-{input_model_name}", type=OUTPUT_DATASET_TYPE)
    artifact.add_dir(save_dir)
    run.log_artifact(artifact)
    run.log(scores)
    run.finish()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
