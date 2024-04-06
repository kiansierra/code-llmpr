import hydra
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from dotenv import load_dotenv
from llm_prompt import FORMATTERS_MAPPING, Example
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import wandb

import wandb
from llm_prompt import FORMATTERS_MAPPING, Example

load_dotenv()

INPUT_DATASET_NAME = "gathered_rewritten_texts"
MODEL_TYPE = "model-sft"

USED_COLUMNS = ["original_text", "rewritten_text", "rewrite_prompt"]

@hydra.main(config_path="llm_prompt/configs/validation", config_name="mistral-7b-it-v2", version_base=None)
def main(config: DictConfig) -> None:
    run = wandb.init(config=OmegaConf.to_container(config), job_type="validation", tags=[config.model_name])
    model_datadir = f"./artifacts/{MODEL_TYPE}"
    model_artifact = run.use_artifact(f"{config.model_name}-sft:latest", type=MODEL_TYPE)
    model_datadir = model_artifact.download(model_datadir)
    config.model.pretrained_model_name_or_path = model_datadir
    quantization_config = BitsAndBytesConfig(**config.quantization)
    model = AutoModelForCausalLM.from_pretrained(
        **config.model, device_map="auto", quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_datadir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    datadir = f"./artifacts/{INPUT_DATASET_NAME}"
    artifact = run.use_artifact(f"{INPUT_DATASET_NAME}:latest")
    datadir = artifact.download(datadir)
    dataset_dict = load_from_disk(datadir)
    test_df_records = dataset_dict['test'].to_pandas()[USED_COLUMNS].to_dict('records')
    message_stack = [Example(**elem) for elem in test_df_records]
    formatter_cls = FORMATTERS_MAPPING[config.formatter.name]
    formatter = formatter_cls(tokenizer,
                              message_stack,
                              system=config.formatter.system,
                              num_examples=config.formatter.num_examples)
    
    validation_dataset = dataset_dict['validation']
    if config.validation_size:
        validation_dataset = validation_dataset.shuffle()
        validation_dataset = validation_dataset.select(range(config.validation_size))
    validation_dataset = validation_dataset.map(
        lambda x, y: {"input": formatter.format_row(x, y)},
        input_columns=["original_text", "rewritten_text"],
        desc="Formatting Inputs",
    )
    rewrite_prompts = validation_dataset['rewrite_prompt']
    validation_dataset = validation_dataset.filter(lambda x: formatter.response_template in x["input"], desc="Filtering Inputs")
    responses = []
    for elem in tqdm(validation_dataset['input'], desc="Generating responses"):
        inputs = tokenizer(elem, return_tensors="pt").to('cuda')
        output = model.generate(**inputs, **config.generation, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        response = response.split(formatter.response_template)[-1].strip()
        responses.append(response)
        
    del model
    torch.cuda.empty_cache()
        
    encoder = SentenceTransformer("sentence-transformers/sentence-t5-base").to("cuda")
    predicted_embeddings = encoder.encode(responses, convert_to_tensor=True)
    prompt_embeddings = encoder.encode(rewrite_prompts, convert_to_tensor=True)
    similarity = F.cosine_similarity(predicted_embeddings, prompt_embeddings)
    results = {"validation/similarity": similarity.mean(), "validation/similarity@3": (similarity**3).mean()}
    run.log(results)
    for k,v in results.items():
        logger.info(f"{k} : {v:.3f}")
    run.finish()
    
    



if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
