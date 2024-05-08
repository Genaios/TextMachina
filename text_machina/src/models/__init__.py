# flake8: noqa
from importlib import import_module
from typing import Mapping

from ..common.exceptions import MissingIntegrationError
from ..config import ModelConfig
from .base import TextGenerationModel

MODELS: Mapping[str, str] = {
    "openai": "OpenAIModel",
    "anthropic": "AnthropicModel",
    "cohere": "CohereModel",
    "hf_local": "HuggingFaceLocalModel",
    "hf_remote": "HuggingFaceRemoteModel",
    "vertex": "VertexModel",
    "bedrock": "BedrockModel",
    "ai21": "AI21Model",
    "azure_openai": "AzureOpenAIModel",
    "inference_server": "InferenceServerModel",
    "open_router": "OpenRouterModel",
    "deep_infra": "DeepInfraModel",
}


def get_model(model_config: ModelConfig) -> TextGenerationModel:
    """
    Gets a text generation model from the pool.

    Args:
        model_config (ModelConfig): a model config.

    Returns:
        TextGenerationModel: a text generation model from the pool.
    """
    provider = model_config.provider
    try:
        model_class = getattr(
            import_module(f".{provider}", __name__),
            MODELS[provider],
        )
    except (ModuleNotFoundError, ImportError):
        raise MissingIntegrationError(integration=provider)
    return model_class(model_config)


__all__ = list(MODELS.values())
