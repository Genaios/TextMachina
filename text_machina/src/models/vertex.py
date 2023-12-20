import os
from typing import Dict

from google.cloud import aiplatform
from google.oauth2 import service_account
from vertexai.preview.language_models import ChatModel as VertexChatModel
from vertexai.preview.language_models import (
    TextGenerationModel as VertexTextGenerationModel,
)

from ..common.logging import get_logger
from ..common.utils import get_instantiation_args
from ..config import ModelConfig
from .base import TextGenerationModel
from .types import GENERATION_ERROR, CompletionType

_logger = get_logger(__name__)


class VertexModel(TextGenerationModel):
    """
    Generates completions using VertexAI models.
    Requires the definition of the `VERTEX_AI_CREDENTIALS_FILE=<path>` environment variable.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

        credentials = service_account.Credentials.from_service_account_file(
            os.environ["VERTEX_AI_CREDENTIALS_FILE"]
        )
        aiplatform.init(
            credentials=credentials,
            **get_instantiation_args(
                aiplatform.init, self.model_config.model_dump()
            ),
        )
        self.model = self._get_model()

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

    def _get_model(self):
        model_class = (
            VertexChatModel
            if self.model_config.api_type == CompletionType.CHAT
            else VertexTextGenerationModel
        )
        return model_class.from_pretrained(self.model_config.model_name)

    def _chat_request(self, prompt: str, generation_config: Dict) -> str:
        return (
            self.model.start_chat()
            .send_message(prompt, **generation_config)
            .text
        )

    def _completion_request(self, prompt: str, generation_config: Dict) -> str:
        return self.model.predict(prompt, **generation_config).text
