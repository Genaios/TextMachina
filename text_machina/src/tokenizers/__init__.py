# flake8: noqa
from importlib import import_module
from typing import Mapping

from ..common.exceptions import MissingIntegrationError
from .base import Tokenizer
from .image import TOKENIZERS as IMAGE_TOKENIZERS
from .text import TOKENIZERS as TEXT_TOKENIZERS

TOKENIZERS: Mapping[str, Mapping[str, str]] = {
    "text": TEXT_TOKENIZERS,
    "image": IMAGE_TOKENIZERS,
}


def get_tokenizer(modality: str, provider: str, model_name: str) -> Tokenizer:
    """
    Gets a tokenizer from the pool.

    Args:
        modality (str): a modality.
        provider (str): a model provider.
        model_name (str): name of a model served by the provider.

    Returns:
        Tokenizer: a tokenizer from the pool.
    """
    tokenizer_cls_name = TOKENIZERS[modality][provider]

    try:
        tokenizer_class = getattr(
            import_module(f".{modality}.{provider}", __name__),
            tokenizer_cls_name,
        )
    except (ModuleNotFoundError, ImportError):
        raise MissingIntegrationError(integration=provider)

    return tokenizer_class(model_name)


__all__ = list(TOKENIZERS.values())
