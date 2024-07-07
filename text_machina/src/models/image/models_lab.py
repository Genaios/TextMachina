import io
import os
from typing import Dict

import requests
from PIL import Image
from requests.adapters import HTTPAdapter, Retry

from ...common.logging import get_logger
from ...config import ModelConfig
from ..base import GenerationModel

_logger = get_logger(__name__)


class ModelsLabModel(GenerationModel[Image.Image]):
    """
    Generates images using ModelLabs community models.

    https://docs.modelslab.com/image-generation/community-models/dreamboothtext2img

    Requires the definition of the `MODELS_LAB_API_KEY=<key>` environment variable.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.client = requests.Session()
        retry_adapter = HTTPAdapter(
            max_retries=Retry(
                total=getattr(self.model_config, "max_retries", 10),
                backoff_factor=getattr(self.model_config, "backoff_factor", 2),
                status_forcelist=[
                    code for code in requests.status_codes._codes if code != 200
                ],
            )
        )
        self.client.mount("http://", retry_adapter)
        self.client.mount("https://", retry_adapter)

    def sample_generate(
        self,
        prompt: str,
        generation_config: Dict,
    ) -> Image.Image:
        payload = {
            "key": os.environ["MODELS_LAB_API_KEY"],
            "model_id": self.model_config.model_name,
            "prompt": prompt,
            **generation_config,
        }

        headers = {"Content-Type": "application/json"}
        try:
            response = self.client.request(
                "POST", url=self.model_config.url, headers=headers, json=payload
            ).json()

            # Fallback from output to future
            output = response.get("output", None)
            if not output:
                output = response.get("proxy_links", None)
                if not output:
                    output = response["future_links"]

            url = output[0]
            return Image.open(io.BytesIO(requests.get(url).content))

        except Exception as e:
            _logger.info(f"Unrecoverable exception during the request: {e}")
            # TODO: We must decide the output type in case of generation errors
            return None
