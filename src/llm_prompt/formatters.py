from typing import Dict, Optional, Protocol

import numpy as np
from transformers import PreTrainedTokenizer

__all__ = ["Formatter", "BaseFormatter", "GemmaITFormatter", "LlamaChatFormatter", "FORMATTERS_MAPPING"]


class Formatter(Protocol):
    def format_row(self, original_text: str, rewritten_text: str, rewrite_prompt: Optional[str]) -> str:
        ...

    @property
    def response_template(self) -> str:
        ...

    @property
    def input_template(self) -> str:
        ...


QUERY_TEMPLATES = [
    "Given the original text: {original_text} \n It has been rewritten to: {rewritten_text}. \n Please provide the prompt used to rewrite the text.",  # noqa: E501
    "### Original Text: {original_text} ### Rewriten Text: {rewritten_text}",
    "Guess what prompt has been used to rewrite the original text: {original_text}\n, Rewritten text: {rewritten_text}",  # noqa: E501
]

SYSTEM_PROMPTS = [
    "You are an LLM trying to predict the prompt used to rewrite the text",
    "You are tasked with predicting the prompt used to rewrite the text",
]


class BaseFormatter(Formatter):
    response_template = "###Prompt Used"
    input_template = "{command}\n{response_template}: {rewrite_prompt}"  # noqa: E501

    def __init__(self, tokenizer: PreTrainedTokenizer, template_index: Optional[int] = None) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.template_index = template_index

    def format_row(self, original_text: str, rewritten_text: str, rewrite_prompt: Optional[str] = None) -> str:
        if self.template_index is not None:
            command = QUERY_TEMPLATES[self.template_index].format(
                original_text=original_text, rewritten_text=rewritten_text
            )
        else:
            command = np.random.choice(QUERY_TEMPLATES).format(
                original_text=original_text, rewritten_text=rewritten_text
            )
        rewrite_prompt = rewrite_prompt or ""
        return self.input_template.format(
            command=command,
            response_template=self.response_template,
            rewrite_prompt=rewrite_prompt,
        )


class ChatFormatter(Formatter):
    response_template = None
    include_system = True

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    def format_row(self, original_text: str, rewritten_text: str, rewrite_prompt: Optional[str] = None) -> str:
        tokenizer = self.tokenizer
        command = np.random.choice(QUERY_TEMPLATES).format(original_text=original_text, rewritten_text=rewritten_text)
        sys_input = np.random.choice(SYSTEM_PROMPTS)
        rewrite_prompt = rewrite_prompt or ""
        chat = [
            {"role": "system", "content": sys_input} if self.include_system else None,
            {"role": "user", "content": command},
            {"role": "assistant", "content": rewrite_prompt},
        ]
        chat = list(filter(bool, chat))
        output = tokenizer.apply_chat_template(chat, tokenize=False)
        output = output.replace(tokenizer.bos_token, "").replace(tokenizer.eos_token, "")
        return output


class LlamaChatFormatter(ChatFormatter):
    response_template = "[/INST]"
    include_system = True

class MistralChatFormatter(ChatFormatter):
    response_template = "[/INST]"
    include_system = False


class GemmaITFormatter(ChatFormatter):
    response_template = "<start_of_turn>model"
    include_system = False


FORMATTERS_MAPPING: Dict[str, type[Formatter]] = {
    "llama": BaseFormatter,
    "base": BaseFormatter,
    "gemma-it": GemmaITFormatter,
    "mistral-it": MistralChatFormatter,
    "llama-chat": LlamaChatFormatter,
}
