
import hydra
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from omegaconf import OmegaConf

RESPONSE_TEMPLATE = "### Prompt Used: "

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['original_text'])):
        text = f"""### Original Text: {example['original_text'][i]} ### Rewriten Text: {example['rewritten_text'][i]} {RESPONSE_TEMPLATE} {example['rewrite_prompt'][i]}"""
        output_texts.append(text)
    return output_texts

@hydra.main(config_path="llm_prompt/configs", config_name="llama2-7b", version_base=None)
def main(config) -> None:
    dataset_dict = load_from_disk("../input/rewritten_texts")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(**config.model, quantization_config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    args = TrainingArguments(**config.trainer)
    lora_config_resolved = OmegaConf.to_container(config.lora, resolve=True)
    lora_config = LoraConfig(**lora_config_resolved)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    collator = DataCollatorForCompletionOnlyLM(RESPONSE_TEMPLATE, tokenizer=tokenizer)
    trainer = SFTTrainer(
        model,
        args,
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['validation'],
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        max_seq_length=1024
    )

    trainer.train()
    
if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter