from abc import ABC, abstractmethod
from typing import Dict, List


class Tokenizer(ABC):
    """
    Base class for tokenizers.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        Encodes a text in token ids.

        Args:
            text (str): a text.

        Returns:
            List[int]: list of token ids.
        """
        ...

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """
        Decodes a list of token ids.

        Args:
            tokens (List[int]): list of token ids.

        Returns:
            text (str): decoded text.
        """
        ...

    def get_token_length(self, text: str) -> int:
        """
        Get the token length of a text.

        Args:
            text (str): a text.

        Returns:
            int: token length of the text.
        """
        return len(self.encode(text))

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """
        Truncates a text to a maximum token length.

        Args:
            text (str): a text.
            max_tokens (int): max token length of the text after truncating.

        Returns:
            str: truncated text.
        """
        return self.decode(self.encode(text)[:max_tokens])

    def truncate_texts(self, texts: List[str], max_tokens: int) -> List[str]:
        """
        Truncates a list of texts to a maximum token length.

        Args:
            texts (List[str]): a list of texts.
            max_tokens (int): max token length of each text after truncating.

        Returns:
            List[str]: list of truncated text.
        """
        return [self.truncate_text(text, max_tokens) for text in texts]

    def distributed_truncate(
        self, texts: Dict[str, List[str]], max_tokens: int
    ) -> Dict[str, List[str]]:
        """
        Truncates texts from different extractors to a maximum token length.
        It distributes the `max_tokens` across all the extractor keys,
        so, when all of them are included in the prompt, they sum at most
        `max_tokens`.

        Example:
            texts = {"summary": ["A", "B"], "headline": ["C", "D"]}
            max_tokens = 256
            output = {
                "summary": [truncated("A", 128), truncated("B", 128)],
                "headline": [truncated("C", 128), truncated("D", 128)],
            }

        Args:
            texts (Dict[str, List[str]]): texts of each extractor.
            max_tokens (int): max length to be distributed across extractors.

        Returns:
            Dict[str, List[str]]: truncated texts of each extractor.
        """
        max_tokens = max_tokens // len(texts)
        for key in texts:
            texts[key] = self.truncate_texts(texts[key], max_tokens)

        return texts
