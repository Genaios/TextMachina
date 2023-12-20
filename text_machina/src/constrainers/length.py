from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np

from .base import Constrainer


class LengthConstrainer(Constrainer, ABC):
    """
    Base class for length constrainers to compute
    automatically the number of tokens to generate.
    """

    def __init__(
        self,
        lengths: List[int],
        provider: str,
        min_tokens: int = 10,
        max_tokens: int = 512,
    ):
        super().__init__()
        self.provider = provider
        self.lengths = lengths
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    @abstractmethod
    def estimate(self) -> Tuple[int, int]:
        """
        Estimates min and max token lengths.

        Returns:
            Tuple[int, int]: tuple with the min and max token lengths.
        """
        ...

    def get_constraints(self) -> Dict[str, int]:
        """
        Obtains min and max token lengths, and returns
        these params according to the provider.

        Returns:
            Dict[str, int]: dict with the token length params.
        """
        min_new_tokens, max_new_tokens = self.estimate()

        if self.provider == "openai":
            return {"max_tokens": max_new_tokens}
        elif self.provider == "anthropic":
            return {"max_tokens_to_sample": max_new_tokens}
        elif self.provider == "cohere":
            return {"max_tokens": max_new_tokens}
        elif self.provider == "vertex":
            return {"max_output_tokens": max_new_tokens}

        return {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
        }


class MeanLengthConstrainer(LengthConstrainer):
    """
    Constrainer within one std radius of the mean of a list of numbers
    """

    def __init__(
        self,
        lengths: List[int],
        provider: str,
        min_tokens: int = 10,
        max_tokens: int = 512,
    ):
        super().__init__(lengths, provider, min_tokens, max_tokens)

    def estimate(self) -> Tuple[int, int]:
        mean = np.mean(self.lengths)
        std = np.std(self.lengths)

        max_new_tokens = int(min(self.max_tokens, mean + 2 * std))
        min_new_tokens = int(max(self.min_tokens, mean - 2 * std))
        return (min_new_tokens, max_new_tokens)


class MedianLengthConstrainer(LengthConstrainer):
    """
    Constrainer within one std radius of the median of a list of numbers
    """

    def __init__(
        self,
        lengths: List[int],
        provider: str,
        min_tokens: int = 10,
        max_tokens: int = 512,
    ):
        super().__init__(lengths, provider, min_tokens, max_tokens)

    def estimate(self) -> Tuple[int, int]:
        median = np.median(self.lengths)
        std = np.std(self.lengths)

        max_new_tokens = int(min(self.max_tokens, median + 2 * std))
        min_new_tokens = int(max(self.min_tokens, median - 2 * std))
        return (min_new_tokens, max_new_tokens)
