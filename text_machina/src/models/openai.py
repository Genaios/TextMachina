import os
from typing import Dict

from openai import OpenAI

from ..common.logging import get_logger
from ..common.utils import get_instantiation_args
from ..config import ModelConfig
from .base import TextGenerationModel
from .types import GENERATION_ERROR, CompletionType

_logger = get_logger(__name__)


class OpenAIModel(TextGenerationModel):
    """
    Generates completions using OpenAI models.

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
            self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model_config.model_name,
                **generation_config,
            )
            .choices[0]
            .message.content
        )

    def _completion_request(self, prompt: str, generation_config: Dict) -> str:
        return (
            self.client.completions.create(
                prompt=prompt,
                model=self.model_config.model_name,
                **generation_config,
            )
            .choices[0]
            .text
        )
