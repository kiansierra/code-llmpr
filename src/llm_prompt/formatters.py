from typing import Dict, Optional, Protocol

from transformers import PreTrainedTokenizer


class Formatter(Protocol):
    def format_row(self, original_text: str, rewritten_text: str, rewrite_prompt: Optional[str]) -> str:
        ...

    def format_batch(self, batch: list[str]) -> list[str]:
        ...

    @property
    def response_template(self) -> str:
        ...

    @property
    def input_template(self) -> str:
        ...


class LlamaFormatter(Formatter):
    response_template = "### Prompt Used: "
    input_template = "### Original Text: {original_text} ### Rewriten Text: {rewritten_text} {response_template} {rewrite_prompt}"  # noqa: E501

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    def format_row(self, original_text: str, rewritten_text: str, rewrite_prompt: Optional[str] = None) -> str:
        rewrite_prompt = rewrite_prompt or ""
        return self.input_template.format(
            original_text=original_text,
            rewritten_text=rewritten_text,
            response_template=self.response_template,
            rewrite_prompt=rewrite_prompt,
        )

    def format_batch(self, batch: Dict[str, list[str]]) -> list[str]:
        outputs = []
        num_sequences = len(batch["original_text"])
        rewrite_prompts = batch.get("rewrite_prompt", [None] * num_sequences)
        for idx in range(num_sequences):
            original_text = batch["original_text"][idx]
            rewritten_text = batch["rewritten_text"][idx]
            rewrite_prompt = rewrite_prompts[idx]
            outputs.append(self.format_row(original_text, rewritten_text, rewrite_prompt))
        return outputs


class GemmaITFormatter(Formatter):
    response_template = "<start_of_turn>model "
    input_template = "<start_of_turn>user ### Original Text: {original_text} ### Rewriten Text: {rewritten_text} <end_of_turn> {response_template} {rewrite_prompt}"  # noqa: E501

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    def format_row(self, original_text: str, rewritten_text: str, rewrite_prompt: Optional[str] = None) -> str:
        rewrite_prompt = rewrite_prompt or ""
        return self.input_template.format(
            original_text=original_text,
            rewritten_text=rewritten_text,
            response_template=self.response_template,
            rewrite_prompt=rewrite_prompt,
        )

    def format_batch(self, batch: Dict[str, list[str]]) -> list[str]:
        outputs = []
        num_sequences = len(batch["original_text"])
        rewrite_prompts = batch.get("rewrite_prompt", [None] * num_sequences)
        for idx in range(num_sequences):
            original_text = batch["original_text"][idx]
            rewritten_text = batch["rewritten_text"][idx]
            rewrite_prompt = rewrite_prompts[idx]
            outputs.append(self.format_row(original_text, rewritten_text, rewrite_prompt))
        return outputs
