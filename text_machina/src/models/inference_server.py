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
    Generates completions using models deployed
    on inference servers like TRT or VLLM. This
    model assumes the default APIs are being used
    (e.g., no OpenAI-compatible API in VLLM)
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

        if self.model_config.inference_server not in ALLOWED_INFERENCE_SERVERS:
            raise InvalidInferenceServer(
                self.model_config.inference_server, ALLOWED_INFERENCE_SERVERS
            )

        self.base_url = self.model_config.base_url
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

    def prepare_data(
        self, prompt: str, generation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepares the json data of a request to the inference server.

        Args:
            prompt (str): a prompt.
            generation_config (Dict[str, Any]): a generation config.

        Returns:
            Dict[str, Any]: json data for a request.
        """
        if self.model_config.inference_server == "vllm":
            return {"text_input": prompt, "parameters": generation_config}
        else:
            return {**{"text_input": prompt}, **generation_config}

    def parse_response(self, response: Dict, prompt_len: int) -> str:
        """
        Get the completion from a response, removing the
        prompt from it if required.

        Args:
            response (Dict): response from the inference server.
            prompt_len (int): len of the prompt.

        Returns:
            str: completion.
        """
        if self.model_config.inference_server == "vllm":
            return response["text_output"][prompt_len:]
        else:
            return response["text_output"][prompt_len:]

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
