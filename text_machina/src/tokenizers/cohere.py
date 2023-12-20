import os
from typing import List

import cohere

from .base import Tokenizer


class CohereTokenizer(Tokenizer):
    """
    Tokenizer for Cohere models.

    Requires the definition of the `COHERE_API_KEY=<key>` environment variable.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = cohere.Client(api_key=os.environ["COHERE_API_KEY"])

    def encode(self, text: str) -> List[int]:
        return self.client.tokenize(text).tokens

    def decode(self, tokens: List[int]) -> str:
        return self.client.detokenize(tokens)
