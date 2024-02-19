import os
from typing import Dict

from ai21 import AI21Client
from ai21.models import ChatMessage

from ..common.logging import get_logger
from ..config import ModelConfig
from .base import TextGenerationModel
from .types import GENERATION_ERROR, CompletionType

_logger = get_logger(__name__)


class AI21Model(TextGenerationModel):
    """
    Generates completions using AI21 models.

    Requires the definition of the `AI21_API_KEY=<api_key>` environment variable.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        api_key = os.environ["AI21_API_KEY"]
        num_retries = getattr(self.model_config, "num_retries", 5)
        timeout_sec = getattr(self.model_config, "timeout_sec", 30)
        self.client = AI21Client(
            api_key=api_key, num_retries=num_retries, timeout_sec=timeout_sec
        )

    def generate_completion(
        self,
        prompt: str,
        generation_config: Dict,
    ) -> str:
        completion_fn = (
            self._chat_request
            if self.model_config.api_type == CompletionType.CHAT
            else self._completion_request
        )
        try:
            completion = completion_fn(prompt, generation_config)
        except Exception as e:
            _logger.info(f"Unrecoverable exception during the request: {e}")
            return GENERATION_ERROR
        return completion

    def _chat_request(self, prompt: str, generation_config: Dict) -> str:
        return (
            self.client.chat.create(
                model=self.model_config.model_name,
                messages=[
                    ChatMessage(
                        text=prompt,
                        role="user",
                    ),
                ],
                system="",
                **generation_config,
            )
            .outputs[0]
            .text
        )

    def _completion_request(self, prompt: str, generation_config: Dict) -> str:
        return (
            self.client.completion.create(
                prompt=prompt,
                model=self.model_config.model_name,
                **generation_config,
            )
            .completions[0]
            .data.text
        )
