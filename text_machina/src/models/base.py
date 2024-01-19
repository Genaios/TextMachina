from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Dict, List

from tqdm import tqdm

from ..config import ModelConfig


class TextGenerationModel(ABC):
    """
    Base class for LLMs.
    """

    def __init__(self, model_config: ModelConfig):
        self.model_config = deepcopy(model_config)

    @abstractmethod
    def generate_completion(
        self,
        prompt: str,
        generation_config: Dict,
    ) -> str:
        """
        Generates a completion for a `prompt` by decoding a model
        parameterized by `generation_config`. This method has to be
        overwritten to implement the completion code.

        Args:
            prompts (str): prompt to generate completions for.
            generation_config (Dict): Dictionary containing the generation parameters.

        Returns:
            str: Generated completion or `.types.GENERATION_ERROR` if there was some error.
        """
        ...

    def generate_completions(
        self,
        prompts: List[str],
        generation_config: Dict,
    ) -> List[str]:
        """Generates a completion for each prompt in a list of `prompts`.

        Args:
            prompts (List[str]): List of prompts to generate completions for.
            generation_config (Dict): Dictionary containing the generation parameters.

        Returns:
            List[str]: List of generated completions.
        """
        completions, responses = [], []
        with ThreadPoolExecutor(
            max_workers=min(self.model_config.threads, len(prompts))
        ) as thread_pool:
            for prompt in prompts:
                responses.append(
                    thread_pool.submit(
                        self.generate_completion, prompt, generation_config
                    )
                )
            # Wait completions
            completions = [response.result() for response in tqdm(responses)]

        return completions
