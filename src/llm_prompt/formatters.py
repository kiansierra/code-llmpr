import copy
from typing import Dict, List, Literal, Optional, Protocol
from pydantic import BaseModel
import numpy as np
from transformers import PreTrainedTokenizer

__all__ = [
    "Formatter",
    "BaseFormatter",
    "GemmaITFormatter",
    "LlamaChatFormatter",
    "FORMATTERS_MAPPING",
    "MessageStackFormatter",
    "MistralChatMessageStackFormatter",
    "MESSAGE_STACK",
    "Example"
]


class Formatter(Protocol):
    def format_row(self, original_text: str, rewritten_text: str, rewrite_prompt: Optional[str]) -> str:
        ...

    @property
    def response_template(self) -> str:
        ...

    @property
    def input_template(self) -> str:
        ...
        
class BaseMessageStackFormatter(Protocol):
    def format_row(self, original_text: str, rewritten_text: str, rewrite_prompt: Optional[str]) -> str:
        ...

    @property
    def response_template(self) -> str:
        ...

    @property
    def input_template(self) -> str:
        ...
        
    @property
    def start_response(self) -> str:
        ...
    
    @property
    def orig_prefix(self) -> str:
        ...

    @property
    def rewrite_prefix(self) -> str:
        ...
        
    @property
    def llm_response_for_rewrite(self) -> str:
        ...



QUERY_TEMPLATES = [
    "Given the original text: {original_text} \n It has been rewritten to: {rewritten_text}. \n Please provide the prompt used to rewrite the text.",
    "### Original Text: {original_text} ### Rewriten Text: {rewritten_text}",
    "Guess what prompt has been used to rewrite the original text: {original_text}\n, Rewritten text: {rewritten_text}",
]

SYSTEM_PROMPTS = [
    "You are an LLM trying to predict the prompt used to rewrite the text",
    "You are tasked with predicting the prompt used to rewrite the text",
]


class BaseFormatter(Formatter):
    response_template = "###Prompt Used"
    input_template = "{command}\n{response_template}: {rewrite_prompt}"

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
        
class Example(BaseModel):
    original_text : str
    rewritten_text : str
    rewrite_prompt : Optional[str] = None
    
    def to_message_list(self, formatter:BaseMessageStackFormatter) -> List[Dict[str, str]]:
        messages = []
        messages.append({"role": "user", "content": f"{formatter.orig_prefix} {self.original_text}"})
        messages.append({"role": "assistant", "content": formatter.llm_response_for_rewrite})
        messages.append({"role": "user", "content": f"{formatter.rewrite_prefix} {self.rewritten_text}"})
        messages.append({"role": "assistant", "content": f"{formatter.start_response} {self.rewrite_prompt}"})
        return messages
    
    


MESSAGE_STACK = [
    {
        "original_text": "William Fletcher (c. 1775–1839), Lord Byron's valet, was often the butt of humour by his famous master.",
        "rewritten_text": "\nIn the annals of time, when tales danced upon the wind, a prophecy foretold of a servant's soul stained with the laughter of a master's jest, a tale etched upon the pages of history. In the realm of whispers and secrets, the name of William Fletcher resonated through the halls of fate, a servant etched upon the hearts of all, the object of both humor and admiration.",
        "rewrite_prompt": "Transform the text into an ancient prophecy.",
    },
    {
        "original_text": 'The Nasura Pillar Site, registered as GcJh3 and also known as Namoratunga II, is an archaeological site on the west side of Lake Turkana in Kenya dating to the Pastoral Neolithic. Namoratunga means "people of stone" in the Turkana language. The site was originally believed to have been created around 300 BC, but recent excavations have yielded an older radiocarbon sample dating to 2398 +/- 44 years BC.',
        "rewritten_text": "\nThe sun shone brightly upon the radiant soil of the Nasura Pillar Site, GcJh3 AKA Namoratunga II, a testament to the enduring spirit of the Pastoral Neolithic that once flourished at this ancient oasis. Towering pillars of stone pierced the sky, whispering tales of a distant era when the world stood on the cusp of change.\n\nThe wind whipped through the air, carrying with it the scent of earth and the echoes of the past",
        "rewrite_prompt": "Style it as a day in the life of a superhero.",
    },
]


class MessageStackFormatter(Formatter):
    start_response = "Prompt Used:"
    orig_prefix = "Original Text:"
    rewrite_prefix = "Re-written Text:"
    llm_response_for_rewrite = "Provide the new text and I will tell you what new element was added or change in tone was made to improve it - with no references to the original.  I will avoid mentioning names of characters.  It is crucial no person, place or thing from the original text be mentioned.  For example - I will not say things like 'change the puppet show into a book report' - I would just say 'improve this text into a book report'.  If the original text mentions a specific idea, person, place, or thing - I will not mention it in my answer.  For example if there is a 'dog' or 'office' in the original text - the word 'dog' or 'office' must not be in my response.  My answer will be a single sentence."
    input_template = "{command}\n{response_template}: {rewrite_prompt}"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        example_stack: List[Example],
        template_index: Optional[int] = None,
        system: Literal['system', 'mock', 'none'] = 'none',
        num_examples: int = 1
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.example_stack = example_stack
        self.template_index = template_index
        self.system = system
        self.num_example_messages = num_examples
        self.message_stack = self.prepare_messages()

    def prepare_messages(self) -> List[Dict[str, str]]:
        messages = []
        if self.system == 'mock':
            messages.append({"role": "user", "content": np.random.choice(SYSTEM_PROMPTS)})
            messages.append({"role": "assistant", "content": "Understood, I will follow your instructions."})
        if self.system == 'system':
            messages.append({"role": "system", "content": np.random.choice(SYSTEM_PROMPTS)})
        chosen_examples = np.random.choice(self.example_stack, self.num_example_messages)
        for example in chosen_examples:
            messages += example.to_message_list(self)
        return messages

    def format_row(self, original_text: str, rewritten_text: str, rewrite_prompt: Optional[str] = None) -> str:
        tokenizer = self.tokenizer
        rewrite_prompt = rewrite_prompt or ""
        chat = self.prepare_messages()
        chat.append({"role": "user", "content": f"{self.orig_prefix} {original_text}"})
        chat.append({"role": "assistant", "content": self.llm_response_for_rewrite})
        chat.append({"role": "user", "content": f"{self.rewrite_prefix} {rewritten_text}"})
        chat.append({"role": "assistant", "content": f"{self.start_response} {rewrite_prompt}"})
        chat = list(filter(bool, chat))
        output = tokenizer.apply_chat_template(chat, tokenize=False)
        output = output.replace(tokenizer.bos_token, "").replace(tokenizer.eos_token, "")
        return output


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


class MistralChatMessageStackFormatter(MessageStackFormatter):
    response_template = f"[/INST]{MessageStackFormatter.start_response}"
    include_system = False
    
class LlamaMesssageStack(MessageStackFormatter):
    response_template = f"[/INST] {MessageStackFormatter.start_response}"
    include_system = False
    
class GemmaMesssageStack(MessageStackFormatter):
    response_template = f"<start_of_turn>model\n{MessageStackFormatter.start_response}"
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
    "mistral-message-stack": MistralChatMessageStackFormatter,
    "llama-message-stack": LlamaMesssageStack,
    "gemma-message-stack": GemmaMesssageStack,
    
}
