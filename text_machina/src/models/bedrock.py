import json
import os
from typing import Any, Dict

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
        generation_config: Dict[str, Any],
    ) -> str:
        request_body = self.get_request_body(prompt, generation_config)

        try:
            response = self.client.invoke_model(
                body=request_body,
                modelId=self.model_config.model_name,
                accept="application/json",
                contentType="application/json",
            )
            response_body = json.loads(response["body"].read())
            completion = self.get_completion_from_response_body(response_body)

        except boto_exceptions.ClientError as ce:
            error_msg = ce.response["Error"]["Code"]
            _logger.info(
                f"Unrecoverable exception during the request: {error_msg}"
            )
            return GENERATION_ERROR
        return completion

    def get_request_body(
        self, prompt: str, generation_config: Dict[str, Any]
    ) -> str:
        """
        Prepares the request body for a request to a bedrock model.

        Considers the different parameters that each model provider accepts.

        Args:
            prompt (str): the prompt to use for generating text.
            generationc_config (Dict[str, Any]): the generation config.
        Returns:
            Dict: a serializable provider-specific request body.
        """
        bedrock_provider = self.model_config.model_name.split(".")[0]
        assert bedrock_provider in {
            "ai21",
            "amazon",
            "anthropic",
            "cohere",
            "meta",
            "mistral",
        }

        if bedrock_provider == "amazon":
            request_body = {
                "inputText": prompt,
                "textGenerationConfig": generation_config,
            }
        elif bedrock_provider == "ai21":
            if "maxTokenCount" in generation_config:
                generation_config["maxTokens"] = generation_config.pop(
                    "maxTokenCount"
                )
            request_body = {"prompt": prompt, **generation_config}
        elif bedrock_provider == "anthropic":
            if "maxTokenCount" in generation_config:
                generation_config["max_tokens_to_sample"] = (
                    generation_config.pop("maxTokenCount")
                )
            request_body = {"prompt": prompt, **generation_config}
        elif bedrock_provider == "cohere":
            # length constrainers work directly on providers themselves
            # so instead we overwrite the key name here
            if "maxTokenCount" in generation_config:
                generation_config["max_tokens"] = generation_config.pop(
                    "maxTokenCount"
                )
            request_body = {"prompt": prompt, **generation_config}
        elif bedrock_provider == "meta":
            if "maxTokenCount" in generation_config:
                generation_config["max_gen_len"] = generation_config.pop(
                    "maxTokenCount"
                )
            request_body = {"prompt": prompt, **generation_config}
        elif bedrock_provider == "mistral":
            if "maxTokenCount" in generation_config:
                generation_config["max_tokens"] = generation_config.pop(
                    "maxTokenCount"
                )
            request_body = {"prompt": prompt, **generation_config}

        return json.dumps(request_body)

    def get_completion_from_response_body(self, response_body: Dict) -> str:
        """
        Obtains the completions from a response body returned by a bedrock model.

        Considers the different API schemas that each model provider uses.

        Args:
            response_body (Dict): the body returned by models in bedrock.
        Returns:
            str: the completion of the model extracted from the body.
        """
        bedrock_provider = self.model_config.model_name.split(".")[0]
        assert bedrock_provider in {
            "ai21",
            "amazon",
            "anthropic",
            "cohere",
            "meta",
            "mistral",
        }

        if bedrock_provider == "amazon":
            completion = response_body["results"][0]["outputText"]
        elif bedrock_provider == "ai21":
            completion = response_body["completions"][0]["data"]["text"]
        elif bedrock_provider == "anthropic":
            completion = response_body["completion"]
        elif bedrock_provider == "cohere":
            completion = response_body["generations"][0]["text"]
        elif bedrock_provider == "meta":
            completion = response_body["generation"]
        elif bedrock_provider == "mistral":
            completion = response_body["outputs"][0]["text"]

        return completion
