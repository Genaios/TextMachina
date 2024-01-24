# flake8: noqa
from importlib import import_module
from typing import Mapping

from ..common.exceptions import MissingIntegrationError
from ..common.logging import get_logger
from .base import Tokenizer

_logger = get_logger(__name__)


TOKENIZERS: Mapping[str, str] = {
    "openai": "OpenAITokenizer",
    "anthropic": "AnthropicTokenizer",
    "cohere": "CohereTokenizer",
    "hf_local": "HuggingFaceLocalTokenizer",
    "hf_remote": "HuggingFaceRemoteTokenizer",
    "vertex": "VertexTokenizer",
    "bedrock": "BedrockTokenizer",
    "ai21": "AI21Tokenizer",
    "azure_openai": "AzureOpenAITokenizer",
    "inference_server": "InferenceServerTokenizer",
}


def get_tokenizer(provider: str, model_name: str) -> Tokenizer:
    """
    Gets a tokenizer from the pool.

    Args:
        provider (str): a model provider.
        model_name (str): name of a model served by the provider.

    Returns:
        Tokenizer: a tokenizer from the pool.
    """
    tokenizer_cls_name = TOKENIZERS[provider]

    if tokenizer_cls_name == "vertex":
        _logger.warn(
            "Vertex does not provide a tokenizer,"
            "OpenAI tokenizer for GPT-4 will be used."
        )

    try:
        tokenizer_class = getattr(
            import_module(f".{provider}", __name__),
            tokenizer_cls_name,
        )
    except (ModuleNotFoundError, ImportError):
        raise MissingIntegrationError(integration=provider)

    return tokenizer_class(model_name)


__all__ = list(TOKENIZERS.values())
