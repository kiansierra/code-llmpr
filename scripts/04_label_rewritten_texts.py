import argparse
import os

import torch
from datasets import load_from_disk
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

import wandb

load_dotenv()
SPLITS = ['train', 'validation', 'test']

INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "../input")
INPUT_DATASET_NAME = "rewritten_texts"
OUTPUT_DATASET_NAME = "labeled_rewritten_texts"

def parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--version", type=int, default=1)
    return argparser.parse_args()

def label_english(model, tokenizer):
    """
    Returns a function that labels the english texts in a batch producint the prob for the english class
    Args:
        model: The model to use for labeling
        tokenizer: The tokenizer to use for tokenizing the texts
    """
    eng_id = model.config.label2id['en']
    
    @torch.no_grad()
    def _label_batch(batch):
        inputs = tokenizer(batch['original_text'], padding=True, truncation=True, return_tensors="pt").to(model.device)
        logits = model(**inputs).logits
        probs = logits.softmax(dim=-1)
        return {'en': probs[:, eng_id]}
    
    return _label_batch

def label_polluted(classifier):
    candidate_labels = ['yes', 'no']
    
    zero_shot_template = "Does the following text contain the information {prompt}? {text}"
    def _batch_predict(batch):
        sequences = []
        
        for promtp, text in zip(batch['rewrite_prompt'], batch['rewritten_text']):
            sequences.append(zero_shot_template.format(prompt=promtp, text=text))
        # Get the prob for the yes class
        outputs = classifier(sequences, candidate_labels)
        prob_yes = [output['scores'][output['labels'].index('yes')] for output in outputs]
        modified_texts = []
        sequences = []
        # Create the sequences for the modified texts, removing the first line likely to contain the prompt information
        for promtp, text in zip(batch['rewrite_prompt'], batch['rewritten_text']):
            if '\n' in text:
                text = "\n".join(text.split('\n')[1:])
            sequences.append(zero_shot_template.format(prompt=promtp, text=text))
            modified_texts.append(text)
        # Get the prob for the yes class on the modified texts
        outputs = classifier(sequences, candidate_labels)
        prob_yes_modified = [output['scores'][output['labels'].index('yes')] for output in outputs]
        final_keep_texts = []
        final_prob_yes = []
        # Keep the text with the lowest prob for the yes class
        for i, (text, prob, prob_modified) in enumerate(zip(modified_texts, prob_yes, prob_yes_modified)):
            if  prob_modified < prob:
                final_keep_texts.append(text)
                final_prob_yes.append(prob_modified)
            else:
                final_keep_texts.append(batch['rewritten_text'][i])
                final_prob_yes.append(prob)
        return {'rewritten_text': final_keep_texts, 'yes': final_prob_yes}
    return _batch_predict

def main(args) -> None:
    run = wandb.init(job_type='label_rewriten_texts', config=vars(args))
    prefix = f'v-{args.version}'
    artifact = run.use_artifact(f"{INPUT_DATASET_NAME}:latest")
    datadir = artifact.download(f'./artifacts/{INPUT_DATASET_NAME}/{prefix}', path_prefix=prefix)
    datasets = load_from_disk(datadir)
    model_ckpt = "papluca/xlm-roberta-base-language-detection"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt).to("cuda")
    datasets = datasets.map(label_english(model, tokenizer),
                            batched=True,
                            batch_size=64,
                            desc="Labeling English texts")
    del model
    torch.cuda.empty_cache()
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device='cuda')
    datasets = datasets.map(label_polluted(classifier),
                            batched=True,
                            batch_size=64,
                            desc="Labeling Polluted responses")
    datasets.save_to_disk(f"{INPUT_DATA_DIR}/{OUTPUT_DATASET_NAME}/{prefix}")
    artifact = wandb.Artifact(OUTPUT_DATASET_NAME, type="dataset")
    artifact.add_dir(f"{INPUT_DATA_DIR}/{OUTPUT_DATASET_NAME}")
    run.log_artifact(artifact)
    run.finish()
    
if __name__ == "__main__":
    args = parser()
    main(args)