import os
from typing import List

from anthropic import Anthropic

from .base import Tokenizer


class AnthropicTokenizer(Tokenizer):
    """
    Tokenizer for Anthropic models.

    Requires the definition of the `ANTRHOPIC_API_KEY=<key>` environment variable.
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.tokenizer = Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        ).get_tokenizer()

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)
