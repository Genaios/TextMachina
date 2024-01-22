import os

from openai import AzureOpenAI

from ..common.utils import get_instantiation_args
from ..config import ModelConfig
from .openai import OpenAIModel


class AzureOpenAIModel(OpenAIModel):
    """
    Generates completions using Azure OpenAI models.

    Requires the definition of the `AZURE_OPENAI_API_KEY=<key>` env variable.
    """

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.model_name = model_config.model_name
        self.client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            **get_instantiation_args(
                AzureOpenAI.__init__, self.model_config.model_dump()
            ),
        )
