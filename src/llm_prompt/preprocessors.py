from typing import Protocol

import einops
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


class Preprocessor(Protocol):
    def setup(self) -> None:
        ...

    def __enter__(self) -> "Preprocessor":
        ...

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        ...

    def cleanup(self) -> None:
        ...

    def __call__(self, text: str) -> str:
        ...


class EnglishLabeler(Preprocessor):
    """
    Creates a function that labels the english texts in a batch producing the prob for the english class
    """

    def preprocess(self, text: str) -> str:
        return text

    def __enter__(self) -> "EnglishLabeler":
        self.setup()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.cleanup()

    def setup(self) -> None:
        model_ckpt = "papluca/xlm-roberta-base-language-detection"
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_ckpt).to("cuda")
        self.eng_id = self.model.config.label2id["en"]

    @torch.no_grad()
    def __call__(self, batch: str) -> str:
        inputs = self.tokenizer(batch["original_text"], padding=True, truncation=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        logits = self.model(**inputs).logits
        probs = logits.softmax(dim=-1)
        return {"en": probs[:, self.eng_id]}

    def cleanup(self) -> None:
        del self.model, self.tokenizer
        torch.cuda.empty_cache()


class ResponsePollutionLabeler(Preprocessor):
    candidate_labels = ["yes", "no"]
    zero_shot_template = "Does the following text contain the information {prompt}? {text}"

    def __init__(self, skip_prob: float = 0.5, prob_original: float = 0.7) -> None:
        super().__init__()
        self.skip_prob = skip_prob
        self.prob_original = prob_original

    def setup(self) -> None:
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device="cuda")

    def cleanup(self) -> None:
        del self.classifier
        torch.cuda.empty_cache()

    def __enter__(self) -> "ResponsePollutionLabeler":
        self.setup()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.cleanup()

    def __call__(self, batch):
        if np.random.rand() < self.skip_prob:
            return {"rewritten_text": batch["rewritten_text"], "yes": [0.0] * len(batch["rewritten_text"])}

        classifier = self.classifier
        zero_shot_template = self.zero_shot_template
        candidate_labels = self.candidate_labels

        sequences = []
        for promtp, text in zip(batch["rewrite_prompt"], batch["rewritten_text"]):
            sequences.append(zero_shot_template.format(prompt=promtp, text=text))
        # Get the prob for the yes class
        outputs = classifier(sequences, candidate_labels)
        prob_yes = [output["scores"][output["labels"].index("yes")] for output in outputs]
        modified_texts = []
        sequences = []
        # Create the sequences for the modified texts, removing the first line likely to contain the prompt information
        for promtp, text in zip(batch["rewrite_prompt"], batch["rewritten_text"]):
            if "\n" in text:
                text = "\n".join(text.split("\n")[1:])
            sequences.append(zero_shot_template.format(prompt=promtp, text=text))
            modified_texts.append(text)
        # Get the prob for the yes class on the modified texts
        outputs = classifier(sequences, candidate_labels)
        prob_yes_modified = [output["scores"][output["labels"].index("yes")] for output in outputs]
        final_keep_texts = []
        final_prob_yes = []
        # Keep the text with the lowest prob for the yes class if original text has prob yes below the threshold
        for i, (text, prob, prob_modified) in enumerate(zip(modified_texts, prob_yes, prob_yes_modified)):
            if prob_modified < prob and prob > self.prob_original:
                final_keep_texts.append(text)
                final_prob_yes.append(prob_modified)
            else:
                final_keep_texts.append(batch["rewritten_text"][i])
                final_prob_yes.append(prob)
        return {"rewritten_text": final_keep_texts, "yes": final_prob_yes}


class CosineScorer(Preprocessor):
    def setup(self):
        self.model = SentenceTransformer("sentence-transformers/sentence-t5-base").to("cuda")

    def cleanup(self):
        del self.model
        torch.cuda.empty_cache()

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.cleanup()

    @staticmethod
    def extend_embeddings(embeddings: torch.Tensor, num_sequences: int) -> torch.Tensor:
        embeddings = einops.repeat(embeddings, "n d -> n m d", m=num_sequences)
        embeddings = einops.rearrange(embeddings, "n m d -> (n m) d")
        return embeddings

    def __call__(self, batch):
        predicted = []
        num_sequences = len(batch["predicted"][0])
        for elem in batch["predicted"]:
            predicted.extend(elem)
        prompts = batch["rewrite_prompt"]
        prompts_embeddings = self.model.encode(prompts, convert_to_tensor=True)
        predicted_embeddings = self.model.encode(predicted, convert_to_tensor=True)
        prompts_embeddings = self.extend_embeddings(prompts_embeddings, num_sequences)
        cosine = F.cosine_similarity(predicted_embeddings, prompts_embeddings)
        cosine_3 = F.cosine_similarity(predicted_embeddings**3, prompts_embeddings**3)
        cosine = einops.rearrange(cosine, "(a b)-> a b", b=num_sequences)
        cosine_3 = einops.rearrange(cosine_3, "(a b)-> a b", b=num_sequences)
        return {"cosine": cosine.tolist(), "cosine_3": cosine_3.tolist()}
