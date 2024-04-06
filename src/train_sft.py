import hydra
import torch
from accelerate import PartialState
from datasets import load_from_disk
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftMixedModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

import wandb
from llm_prompt import FORMATTERS_MAPPING, MESSAGE_STACK

load_dotenv()

INPUT_DATASET_NAME = "gathered_rewritten_texts"
MODEL_OUTPUT_TYPE = "model-sft"


@hydra.main(config_path="llm_prompt/configs/sft", config_name="mistral-7b-it-v2", version_base=None)
def main(config: DictConfig) -> None:
    state = PartialState()
    quantization_config = BitsAndBytesConfig(**config.quantization)
    model = AutoModelForCausalLM.from_pretrained(
        **config.model, device_map={"": state.process_index}, quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    formatter = FORMATTERS_MAPPING[config.formatter](tokenizer, MESSAGE_STACK)
    datadir = f"./artifacts/{INPUT_DATASET_NAME}"
    if state.is_main_process:
        run = wandb.init(config=OmegaConf.to_container(config), job_type="train_sft", tags=[config.model_name])
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
    dataset_dict = dataset_dict.map(
        lambda x, y, z: {"input": formatter.format_row(x, y, z)},
        input_columns=["original_text", "rewritten_text", "rewrite_prompt"],
        desc="Formatting Inputs",
    )
    dataset_dict = dataset_dict.select_columns(["input"])
    dataset_dict = dataset_dict.filter(lambda x: formatter.response_template in x["input"], desc="Filtering Inputs")

    collator = DataCollatorForCompletionOnlyLM(formatter.response_template, tokenizer=tokenizer)
    trainer = SFTTrainer(
        model,
        args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        dataset_text_field="input",
        data_collator=collator,
        max_seq_length=2048,
    )

    trainer.train()
    if state.is_main_process:
        tokenizer.save_pretrained(config.trainer.output_dir)
        model.save_pretrained(f"{config.trainer.output_dir}/last")
        del model
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(**config.model, device_map={"": state.process_index})
        lora_model = PeftMixedModel.from_pretrained(model, f"{config.trainer.output_dir}/last")
        merged_model = lora_model.merge_and_unload()
        merged_model.save_pretrained(config.trainer.output_dir, safe_serialization=True, max_shard_size="4GB")
        OmegaConf.save(config, f"{config.trainer.output_dir}/config.yaml")
        artifact = wandb.Artifact(f"{config.model_name}-sft", type=MODEL_OUTPUT_TYPE)
        artifact.add_dir(config.trainer.output_dir)
        run.log_artifact(artifact)
        run.finish()
    state.wait_for_everyone()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
