# flake8: noqa
from typing import Mapping

TOKENIZERS: Mapping[str, str] = {
    "openai": "OpenAITokenizer",
    "anthropic": "AnthropicTokenizer",
    "cohere": "CohereTokenizer",
    "hf_local": "HuggingFaceLocalTokenizer",
    "hf_remote": "HuggingFaceRemoteTokenizer",
    "vertex": "VertexTokenizer",
    "bedrock": "BedrockTokenizer",
    "ai21": "AI21Tokenizer",
    "azure_openai": "AzureOpenAITokenizer",
    "inference_server": "InferenceServerTokenizer",
    "open_router": "OpenRouterTokenizer",
    "deep_infra": "DeepInfraTokenizer",
    "models_lab": "ModelsLabTokenizer",
}
