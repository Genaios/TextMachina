import os
from typing import Dict

from cohere import Client

from ..common.logging import get_logger
from ..common.utils import get_instantiation_args
from ..config import ModelConfig
from .base import TextGenerationModel
from .types import GENERATION_ERROR

_logger = get_logger(__name__)


class CohereModel(TextGenerationModel):
    """
    Generates completions using Cohere models.

    Requires the definition of the `COHERE_API_KEY=<key>` environment variable.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.client = Client(
            api_key=os.environ["COHERE_API_KEY"],
            **get_instantiation_args(
                Client.__init__, self.model_config.model_dump()
            ),
        )

    def generate_completion(
        self,
        prompt: str,
        generation_config: Dict,
    ) -> str:
        try:
            completion = (
                self.client.generate(
                    model=self.model_config.model_name,
                    prompt=prompt,
                    **generation_config,
                )
                .generations[0]
                .text
            )
        except Exception as e:
            _logger.info(f"Unrecoverable exception during the request: {e}")
            return GENERATION_ERROR
        return completion
