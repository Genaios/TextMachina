from typing import List

from transformers import AutoTokenizer

from .base import Tokenizer


class HuggingFaceRemoteTokenizer(Tokenizer):
    """
    Tokenizer for HuggingFace models served in Inference API or Endpoints.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
