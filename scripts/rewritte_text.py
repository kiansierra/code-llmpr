from datasets import load_from_disk, DatasetDict
from llm_prompt import GemmaGenerator, REWRITE_TEMPLATE

def main():
    dd = load_from_disk('../input/templates')
    dd = dd.map(lambda x: {'input': REWRITE_TEMPLATE.format(**x)}, desc="Rewriting prompts")
    VARIANT = "7b-it-quant"
    weights_dir = '../checkpoints/7b-it-quant'
    
    
    generator = GemmaGenerator(VARIANT, weights_dir, {})
    generator.setup()
    dd = dd.shuffle(42)
    dataset_dict = {}
    for key, dataset in dd.items():
        dataset = dataset.select(range(min(500, len(dataset))))
        dataset = dataset.map(generator.generate_batch,
                              batched=True,
                              batch_size=4,
                              input_columns=['input'],
                              desc=f"Generating rewritten text for {key}")
        dataset_dict[key] = dataset
    dd = DatasetDict(dataset_dict)
    dd.save_to_disk('../input/rewritten_texts')
    # dd.push_to_hub("ksmcg/rewritten_texts", private=True)
    
if __name__ == "__main__":
    main()