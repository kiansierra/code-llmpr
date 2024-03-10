from datasets import load_from_disk, DatasetDict
from llm_prompt import GemmaGenerator

def main():
    dd = load_from_disk('../inputs/templates')
    dd = dd.map(lambda x: {'input': x['rewrite_prompt'].format(x['original_text'])}, desc="Rewriting prompts")
    VARIANT = "7b-it-quant" 
    weights_dir = '../checkpoints/7b-it-quant' 
    
    
    generator = GemmaGenerator(VARIANT, weights_dir, {})
    generator.setup()
    dd = dd.shuffle(42)
    dataset_dict = {}
    for key, dataset in dd.items():
        dataset = dataset.select(range(1000))
        dataset = dataset.map(generator.generate_batch, 
                              batched=True,
                              batch_size=4,
                              input_columns=['input'],
                              desc=f"Generating rewritten text for {key}")
        dataset_dict[key] = dd
    dd = DatasetDict(dataset_dict)
    dd.save_to_disk('../inputs/rewritten_texts')
    
if __name__ == "__main__":
    main()