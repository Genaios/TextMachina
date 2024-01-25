import json
import os
from typing import Dict

import boto3
from botocore import exceptions as boto_exceptions
from botocore.config import Config as BotoConfig

from ..common.logging import get_logger
from ..common.utils import get_instantiation_args
from ..config import ModelConfig
from .base import TextGenerationModel
from .types import GENERATION_ERROR

_logger = get_logger(__name__)


class BedrockModel(TextGenerationModel):
    """
    Generates completions using AWS Bedrock models.

    Requires the definition of the `AWS_ACCESS_KEY_ID=<key>` and
    `AWS_SECRET_ACCESS_KEY=<key>` environment variables.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        client_config = BotoConfig(
            **get_instantiation_args(
                BotoConfig.__init__,
                self.model_config.model_dump(),
                accepted_params=list(BotoConfig.OPTION_DEFAULTS.keys()),
            )
        )
        self.client = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            config=client_config,
        )

    def generate_completion(
        self,
        prompt: str,
        generation_config: Dict,
    ) -> str:
        body_request = json.dumps(
            {"inputText": prompt, "textGenerationConfig": generation_config}
        )
        try:
            response = self.client.invoke_model(
                body=body_request,
                modelId=self.model_config.model_name,
                accept="application/json",
                contentType="application/json",
            )
            response_body = json.loads(response["body"].read())
            completion = response_body["results"][0]["outputText"]

        except boto_exceptions.ClientError as ce:
            error_msg = ce.response["Error"]["Code"]
            _logger.info(
                f"Unrecoverable exception during the request: {error_msg}"
            )
            return GENERATION_ERROR
        return completion
