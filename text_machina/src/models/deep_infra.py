import os

from openai import OpenAI

from ..common.utils import get_instantiation_args
from ..config import ModelConfig
from .openai import OpenAIModel


class DeepInfraModel(OpenAIModel):
    """
    Generates completions using DeepInfra models using the OpenAI interface.

    Requires the definition of the `DEEP_INFRA_API_KEY=<key>` env variable.
    """

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.model_name = model_config.model_name
        self.client = OpenAI(
            api_key=os.environ["DEEP_INFRA_API_KEY"],
            **get_instantiation_args(
                OpenAI.__init__, self.model_config.model_dump()
            ),
        )
