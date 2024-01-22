from typing import Any, Dict

import requests
from requests.adapters import HTTPAdapter, Retry

from ..common.exceptions import InvalidInferenceServer
from ..common.logging import get_logger
from ..config import ModelConfig
from .base import TextGenerationModel
from .types import GENERATION_ERROR

_logger = get_logger(__name__)


ALLOWED_INFERENCE_SERVERS = ["vllm", "trt"]


class InferenceServerModel(TextGenerationModel):
    """
    Generates completions using models deployed on
    TRT and VLLM inference servers.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.client = requests.Session()
        retry_adapter = HTTPAdapter(
            max_retries=Retry(
                total=getattr(self.model_config, "max_retries", 5),
                backoff_factor=getattr(
                    self.model_config, "backoff_factor", 0.5
                ),
                status_forcelist=[
                    code for code in requests.status_codes._codes if code != 200
                ],
            )
        )
        self.client.mount("http://", retry_adapter)
        self.client.mount("https://", retry_adapter)

    def prepare_data(
        self, prompt: str, generation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.model_config.inference_server == "vllm":
            return {"text_input": prompt, "parameters": generation_config}
        elif self.model_config.inference_server == "trt":
            return {**{"text_input": prompt}, **generation_config}
        else:
            raise InvalidInferenceServer(
                self.model_config.inference_server, ALLOWED_INFERENCE_SERVERS
            )

    def parse_response(self, response: Dict, prompt_len: int) -> str:
        if self.model_config.inference_server == "vllm":
            return response["text_output"][prompt_len:]
        elif self.model_config.inference_server == "trt":
            return response["text_output"][prompt_len:]
        else:
            raise InvalidInferenceServer(
                self.model_config.inference_server, ALLOWED_INFERENCE_SERVERS
            )

    def generate_completion(
        self,
        prompt: str,
        generation_config: Dict,
    ) -> str:
        json_data = self.prepare_data(prompt, generation_config)
        try:
            response = self.client.post(self.base_url, json=json_data).json()
            completion = self.parse_response(response, len(prompt))
        except Exception as e:
            _logger.info(f"Unrecoverable exception during the request: {e}")
            return GENERATION_ERROR
        return completion
