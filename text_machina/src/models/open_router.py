import os
from typing import Dict

import requests
from requests.adapters import HTTPAdapter, Retry

from ..common.logging import get_logger
from ..config import ModelConfig
from .base import TextGenerationModel
from .types import GENERATION_ERROR

_logger = get_logger(__name__)


class OpenRouterModel(TextGenerationModel):
    """
    Generates completions using HuggingFace's models remotely deployed
    (HuggingFace's Inference API or Inference Endpoints).

    Requires the definition of the `OPENROUTER_API_KEY=<token>` environment variable.
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

    def generate_completion(self, prompt: str, generation_config: Dict) -> str:
        headers = {
            "Authorization": f'Bearer {os.environ["OPENROUTER_API_KEY"]}'
        }

        payload = {
            "model": self.model_config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            **generation_config,
        }

        try:
            response = self.client.post(
                self.model_config.url, headers=headers, json=payload
            )
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            _logger.info(f"Unrecoverable exception during the request: {e}")
            return GENERATION_ERROR
