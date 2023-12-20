from typing import List, Mapping, Type

from ..tokenizers import get_tokenizer
from .length import (
    LengthConstrainer,
    MeanLengthConstrainer,
    MedianLengthConstrainer,
)

LENGTH_CONSTRAINERS: Mapping[str, Type[LengthConstrainer]] = {
    "mean_length": MeanLengthConstrainer,
    "median_length": MedianLengthConstrainer,
}


def get_length_constrainer(
    texts: List[str],
    model_name: str,
    provider: str,
    constrainer_name: str = "mean_length",
    min_tokens: int = 10,
    max_tokens: int = 512,
) -> LengthConstrainer:
    """
    Returns a length constrainer given by the tokenized lengths of the texts
    """
    tokenizer = get_tokenizer(provider, model_name)
    # Clip texts to `max_tokens`
    lengths = list(
        map(lambda x: min(len(tokenizer.encode(x)), max_tokens), texts)
    )
    constrainer = LENGTH_CONSTRAINERS[constrainer_name]
    return constrainer(
        lengths, provider, min_tokens=min_tokens, max_tokens=max_tokens
    )


__all__ = [str(cls) for cls in LENGTH_CONSTRAINERS.values()]
