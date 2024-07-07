import io
import os
from typing import Dict

import requests
from openai import OpenAI
from PIL import Image

from ...common.logging import get_logger
from ...common.utils import get_instantiation_args
from ...config import ModelConfig
from ..base import GenerationModel

_logger = get_logger(__name__)


class OpenAIModel(GenerationModel[Image.Image]):
    """
    Generates images using OpenAI models.

    https://platform.openai.com/docs/guides/images/usage

    Requires the definition of the `OPENAI_API_KEY=<key>` environment variable.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            **get_instantiation_args(
                OpenAI.__init__, self.model_config.model_dump()
            ),
        )

    def sample_generate(
        self,
        prompt: str,
        generation_config: Dict,
    ) -> Image.Image:

        try:
            response = self.client.images.generate(
                model=self.model_config.model_name,
                prompt=prompt,
                **generation_config,
            )
            return Image.open(
                io.BytesIO(requests.get(response.data[0].url).content)
            )

        except Exception as e:
            _logger.info(f"Unrecoverable exception during the request: {e}")
            # TODO: We must decide the output type in case of generation errors
            return None
