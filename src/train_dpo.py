import hydra
import numpy as np
import torch
from accelerate import PartialState
from datasets import Dataset, load_from_disk
from dotenv import load_dotenv
from llm_prompt import FORMATTERS_MAPPING
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, DPOTrainer, SFTTrainer

import wandb

import wandb
from llm_prompt import FORMATTERS_MAPPING

load_dotenv()

DTYPE_MAPPING = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}

INPUT_DATASET_TYPE = "generated_and_scored_texts"

OmegaConf.register_new_resolver("dtype", lambda x: DTYPE_MAPPING[x])

def convert_dataset_to_dpo(dataset:Dataset) -> Dataset:
    df = dataset.to_pandas()
    df['id'] = df.index
    df['best'] = df['cosine'].apply(lambda x: np.array(x).argmax())
    df['chosen'] = df.apply(lambda x: x['predicted'][x['best']], axis=1)
    df = df.explode('predicted')
    df['count'] = df.groupby((df['id'] != df['id'].shift(1)).cumsum()).cumcount()
    df = df.query("count != best").reset_index(drop=True)
    df = df.rename(columns={'predicted': 'rejected'})
    return Dataset.from_pandas(df)


@hydra.main(config_path="llm_prompt/configs/dpo", config_name="llama2-7b-chat-dpo", version_base=None)
def main(config: DictConfig) -> None:
    state = PartialState()
    quantization_config = BitsAndBytesConfig(**config.quantization)
    model = AutoModelForCausalLM.from_pretrained(
        **config.model, device_map={"": state.process_index}, quantization_config=quantization_config
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    formatter = FORMATTERS_MAPPING[config.formatter](tokenizer)
    datadir = f"./artifacts/{INPUT_DATASET_TYPE}"
    input_model_name = config.model_name
    dataset_name = 'v-5'
    if state.is_main_process:
        run = wandb.init(config=OmegaConf.to_container(config), job_type="train_dpo")
        artifact = run.use_artifact(f"{dataset_name}-{input_model_name}:latest", type=INPUT_DATASET_TYPE)
        datadir = artifact.download(datadir)
    state.wait_for_everyone()
    dataset_dict = load_from_disk(datadir)
    modeldir = f"./artifacts/{config.model_name}"
    if state.is_main_process:
        artifact = run.use_artifact(f"{config.model_name}:latest")
        modeldir = artifact.download(modeldir)
    state.wait_for_everyone()
    
    model = PeftModel.from_pretrained(
        model,
        modeldir,
        is_trainable=True,
        adapter_name="train_adapter",
    )
    model.load_adapter(modeldir, adapter_name="reference")

    args = TrainingArguments(**config.trainer)
    # model = prepare_model_for_kbit_training(
    #     model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
    # )
    dataset_dict["train"] = convert_dataset_to_dpo(dataset_dict["train"])
    
    def process(row):
        chosen = formatter.format_row(row["original_text"], row["rewritten_text"], row["chosen"])
        rejected = formatter.format_row(row["original_text"], row["rewritten_text"], row["rejected"])
        prompt = formatter.format_row(row["original_text"], row["rewritten_text"], None)
        return {"chosen": chosen, "rejected": rejected, "prompt": prompt}
    
    dataset_dict["train"] = dataset_dict["train"].map(process).select_columns(["chosen", "rejected", "prompt"])
        
    trainer = DPOTrainer(
        model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=dataset_dict["train"],
        model_adapter_name="train_adapter",
        ref_adapter_name="reference",
    )

    trainer.train()
    if state.is_main_process:
        model.save_pretrained(config.trainer.output_dir)
        OmegaConf.save(config, f"{config.trainer.output_dir}/config.yaml")
        artifact = wandb.Artifact(config.model_name, type="model")
        artifact.add_dir(config.trainer.output_dir)
        run.log_artifact(artifact)
        run.finish()
    state.wait_for_everyone()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
