import os
import random
import time
from typing import Dict

import requests

from ..common.logging import get_logger
from ..config import ModelConfig
from .base import TextGenerationModel
from .types import GENERATION_ERROR

_logger = get_logger(__name__)


def backoff_with_jitter(retry, cap=600, base=2):
    """Exponential backoff with Jitter"""
    return random.randrange(0, min(cap, base * 2**retry)) + random.random()


class HuggingFaceRemoteModel(TextGenerationModel):
    """
    Generates completions using HuggingFace's models remotely deployed
    (HuggingFace's Inference API or Inference Endpoints).
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.max_retries = getattr(self.model_config, "max_retries", 5)

    def generate_completion(self, prompt: str, generation_config: Dict) -> str:
        headers = {"Authorization": f'Bearer {os.environ["HF_TOKEN"]}'}
        payload = {
            "inputs": prompt,
            "parameters": generation_config,
        }
        retry = 1
        while retry <= self.max_retries:
            try:
                response = requests.post(
                    self.model_config.url, headers=headers, json=payload
                )
            except Exception as e:
                _logger.info(f"Unrecoverable exception during the request: {e}")
                return GENERATION_ERROR
            else:
                # Success
                if response.status_code == 200:
                    return response.json()[0]["generated_text"]
                # Recoverable error
                elif response.status_code in [429, 503]:
                    wait_time = backoff_with_jitter(retry)
                    _logger.info(f"Recoverable error: {response.text}.")
                    _logger.info(
                        f"Waiting {wait_time:.2f}s and retrying"
                        f" up to {self.max_retries} times..."
                    )
                    time.sleep(wait_time)
                    retry += 1
                # Unrecoverable error
                else:
                    _logger.info(
                        f"Request returned an unrecoverable error:"
                        f" {response.status_code}, {response.text}"
                    )
                    return GENERATION_ERROR
        return GENERATION_ERROR
