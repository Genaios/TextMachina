from typing import List

from transformers import AutoTokenizer

from .base import Tokenizer


class HuggingFaceLocalTokenizer(Tokenizer):
    """
    Tokenizer for HuggingFace models.
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
