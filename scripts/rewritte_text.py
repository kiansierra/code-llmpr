from datasets import DatasetDict, load_from_disk, concatenate_datasets
import shutil
from llm_prompt import REWRITE_TEMPLATE, GemmaGenerator


def main():
    dd = load_from_disk('../input/templates')
    existing_templates = load_from_disk('../input/rewritten_texts')
    dd = dd.map(lambda x: {'input': REWRITE_TEMPLATE.format(**x)}, desc="Rewriting prompts", num_proc=4)
    VARIANT = "7b-it-quant"
    weights_dir = '../checkpoints/7b-it-quant'
    
    
    generator = GemmaGenerator(VARIANT, weights_dir, {})
    generator.setup()
    dd = dd.shuffle(42)
    dataset_dict = {}
    for key, dataset in dd.items():
        existing_dataset = existing_templates[key]
        dataset = dataset.filter(lambda x: x['input'] not in existing_dataset['input'],  num_proc=12)
        dataset = dataset.select(range(min(100, len(dataset))))
        dataset = dataset.map(generator.generate_batch,
                              batched=True,
                              batch_size=4,
                              input_columns=['input'],
                              desc=f"Generating rewritten text for {key}")
        dataset_dict[key] = concatenate_datasets([dataset, existing_dataset])
    dd = DatasetDict(dataset_dict)
    dd.save_to_disk('../input/rewritten_texts')
    # dd.push_to_hub("ksmcg/rewritten_texts", private=True)
    
if __name__ == "__main__":
    main()