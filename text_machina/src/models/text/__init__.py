# flake8: noqa
from typing import Mapping

MODELS: Mapping[str, str] = {
    "openai": "OpenAIModel",
    "anthropic": "AnthropicModel",
    "cohere": "CohereModel",
    "hf_local": "HuggingFaceLocalModel",
    "hf_remote": "HuggingFaceRemoteModel",
    "vertex": "VertexModel",
    "bedrock": "BedrockModel",
    "ai21": "AI21Model",
    "azure_openai": "AzureOpenAIModel",
    "inference_server": "InferenceServerModel",
    "open_router": "OpenRouterModel",
    "deep_infra": "DeepInfraModel",
}
