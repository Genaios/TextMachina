from typing import List

import tiktoken

from .base import Tokenizer


class AzureOpenAITokenizer(Tokenizer):
    """
    Tokenizer for AzureOpenAI models.
    Tokenizer can't be inferred from the deployment name
    of a model. GPT-4 Tokenizer is used instead.
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)
