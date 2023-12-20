import os
from typing import Dict

from anthropic import Anthropic

from ..common.logging import get_logger
from ..common.utils import get_instantiation_args
from ..config import ModelConfig
from .base import TextGenerationModel
from .types import GENERATION_ERROR

_logger = get_logger(__name__)


class AnthropicModel(TextGenerationModel):
    """
    Generates completions using Anthropic models.

    Requires the definition of the `ANTRHOPIC_API_KEY=<key>` environment variable.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.client = Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            **get_instantiation_args(
                Anthropic.__init__, self.model_config.model_dump()
            ),
        )

    def generate_completion(
        self,
        prompt: str,
        generation_config: Dict,
    ) -> str:
        try:
            completion = self.client.completions.create(
                model=self.model_config.model_name,
                prompt=prompt,
                **generation_config,
            ).completion
        except Exception as e:
            _logger.info(f"Unrecoverable exception during the request: {e}")
            return GENERATION_ERROR
        return completion
