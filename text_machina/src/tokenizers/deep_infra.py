from typing import List

import tiktoken

from .base import Tokenizer


class DeepInfraTokenizer(Tokenizer):
    """
    Tokenizer for DeepInfra models.

    DeepInfra does not offer tokenizers. GPT-4 Tokenizer is used instead.
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)
