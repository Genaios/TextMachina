import os
from typing import List

import ai21

from .base import Tokenizer


class AI21Tokenizer(Tokenizer):
    """
    Tokenizer for AI21 models.

    Requires the definition of the `AI21_API_KEY=<key>` environment variable.
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)

    def encode(self, text: str) -> List[int]:
        tokenized = ai21.Tokenization.execute(
            api_key=os.environ["AI21_API_KEY"], model=self.model_name, text=text
        )
        return [token["token"] for token in tokenized["tokens"]]

    def decode(self, tokens: List[int]) -> str:
        return "".join([str(token).replace("▁", " ") for token in tokens])[1:]
