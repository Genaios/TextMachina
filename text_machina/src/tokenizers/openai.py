from typing import List

import tiktoken

from .base import Tokenizer


class OpenAITokenizer(Tokenizer):
    """
    Tokenizer for OpenAI models.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)
