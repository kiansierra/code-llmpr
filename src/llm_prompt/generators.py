import contextlib
import os
from typing import Protocol

import requests
import torch
from gemma.config import get_config_for_2b, get_config_for_7b
from gemma.model import GemmaForCausalLM

__all__ = ["BaseGenerator", "GemmaGenerator", "APIGenerator"]


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)


class BaseGenerator(Protocol):
    def setup(self) -> None:
        ...

    def generate(self, texts: str | list[str]):
        ...

    def process_batch(self, texts: list[str]):
        ...

    @property
    def model_id(self):
        ...


class GemmaGenerator(BaseGenerator):
    def __init__(self, variant: str, weights_dir: str, generation_params: dict):
        self.variant = variant
        self.weights_dir = weights_dir
        self.generation_params = generation_params

    @property
    def model_id(self):
        return self.variant

    def setup(self) -> None:
        model_config = get_config_for_2b() if "2b" in self.variant else get_config_for_7b()
        model_config.tokenizer = os.path.join(self.weights_dir, "tokenizer.model")
        model_config.quant = "quant" in self.variant
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with _set_default_tensor_type(model_config.get_dtype()):
            model = GemmaForCausalLM(model_config)
            ckpt_path = os.path.join(self.weights_dir, f"gemma-{self.variant}.ckpt")
            model.load_weights(ckpt_path)
            model = model.to(self.device).eval()
        self.model = model

    def generate(self, texts: str | list[str]):
        outputs = self.model.generate(texts, self.device, **self.generation_params)
        return outputs

    def generate_batch(self, texts: list[str]):
        rewritten_texts = self.generate(texts)
        outputs = {"rewritten_text": rewritten_texts, "model": [self.model_id] * len(texts)}
        generation_params = {k: [v] * len(texts) for k, v in self.generation_params.items()}
        outputs = {**outputs, **generation_params}
        return outputs


class APIGenerator(BaseGenerator):
    def __init__(self, variant: str, timeout: int = 10):
        self.variant = variant
        self.timeout = timeout

    @property
    def model_id(self):
        return self.variant

    def setup(self) -> None:
        self.endpoint = f"https://api-inference.huggingface.co/models/{self.variant}"
        api_token = os.environ.get("HF_API_TOKEN", None)
        if api_token is None:
            raise ValueError("Please set the HF_API_TOKEN environment variable")
        self.headers = {"Authorization": f"Bearer {api_token}"}

    def generate(self, texts: str | list[str]):
        outputs = []
        for text in texts:
            response = requests.post(self.endpoint, headers=self.headers, json={"inputs": text}, timeout=self.timeout)
            if response.status_code != 200:
                outputs.append("EMPTY")
                continue
            generated_text = response.json()[0]["generated_text"]
            generated_text = generated_text.replace(text, "")
            outputs.append(generated_text)
        return outputs

    def generate_batch(self, texts: list[str]):
        rewritten_texts = self.generate(texts)
        outputs = {"rewritten_text": rewritten_texts, "model": [self.model_id] * len(texts)}
        return outputs
