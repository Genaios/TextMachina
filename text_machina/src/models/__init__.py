# flake8: noqa
from importlib import import_module

from ..common.exceptions import MissingIntegrationError
from ..config import ModelConfig
from .base import GenerationModel
from .image import MODELS as IMAGE_MODELS
from .text import MODELS as TEXT_MODELS

MODELS = {"text": TEXT_MODELS, "image": IMAGE_MODELS}


def get_model(modality: str, model_config: ModelConfig) -> GenerationModel:
    """
    Gets a generation model from the pool.

    Args:
        model_config (ModelConfig): a model config.

    Returns:
        GenerationModel: a generation model from the pool.
    """
    provider = model_config.provider
    try:
        model_class = getattr(
            import_module(f".{modality}.{provider}", __name__),
            MODELS[modality][provider],
        )
    except (ModuleNotFoundError, ImportError):
        raise MissingIntegrationError(integration=provider)
    return model_class(model_config)
