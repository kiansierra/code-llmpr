import hydra
import torch
from accelerate import PartialState
from datasets import load_from_disk
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

import wandb
from llm_prompt import FORMATTERS_MAPPING

load_dotenv()

DTYPE_MAPPING = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}

INPUT_DATASET_NAME = "gathered_rewritten_texts"

OmegaConf.register_new_resolver("dtype", lambda x: DTYPE_MAPPING[x])


@hydra.main(config_path="llm_prompt/configs", config_name="llama2-7b-chat", version_base=None)
def main(config: DictConfig) -> None:
    state = PartialState()
    quantization_config = BitsAndBytesConfig(**config.quantization)
    model = AutoModelForCausalLM.from_pretrained(
        **config.model, device_map={"": state.process_index}, quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    formatter = FORMATTERS_MAPPING[config.formatter](tokenizer)
    datadir = f"./artifacts/{INPUT_DATASET_NAME}"
    if state.is_main_process:
        run = wandb.init(config=OmegaConf.to_container(config), job_type="train_sft")
        artifact = run.use_artifact(f"{INPUT_DATASET_NAME}:latest")
        datadir = artifact.download(datadir)
    state.wait_for_everyone()
    dataset_dict = load_from_disk(datadir)

    args = TrainingArguments(**config.trainer)
    lora_config_resolved = OmegaConf.to_container(config.lora, resolve=True)
    lora_config = LoraConfig(**lora_config_resolved)
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = get_peft_model(model, lora_config)

    collator = DataCollatorForCompletionOnlyLM(formatter.response_template, tokenizer=tokenizer)
    trainer = SFTTrainer(
        model,
        args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        formatting_func=formatter.format_batch,
        data_collator=collator,
        max_seq_length=1024,
    )

    trainer.train()
    if state.is_main_process:
        OmegaConf.save(config, f"{config.trainer.output_dir}/config.yaml")
        artifact = wandb.Artifact(config.model_name, type="model")
        artifact.add_dir(config.trainer.output_dir)
        run.log_artifact(artifact)
        run.finish()
    state.wait_for_everyone()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
