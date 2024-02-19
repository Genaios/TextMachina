from typing import List

from ai21_tokenizer import Tokenizer as AITokenizer

from .base import Tokenizer


class AI21Tokenizer(Tokenizer):
    """
    Tokenizer for AI21 models.

    Requires the definition of the `AI21_API_KEY=<key>` environment variable.
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.tokenizer = AITokenizer.get_tokenizer()

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)
