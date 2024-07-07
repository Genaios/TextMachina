from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Dict, Generic, List, TypeVar

from tqdm import tqdm

from ..config import ModelConfig

T = TypeVar("T")


class GenerationModel(ABC, Generic[T]):
    """
    Base class for LLMs.
    """

    def __init__(self, model_config: ModelConfig):
        self.model_config = deepcopy(model_config)

    @abstractmethod
    def sample_generate(
        self,
        prompt: str,
        generation_config: Dict,
    ) -> T:
        """
        Generates by prompting with `prompt` a generation model
        parameterized by `generation_config`. This method has to be
        overwritten to implement the completion code.

        Args:
            prompt (str): prompt to generate.
            generation_config (Dict): generation parameters

        Returns:
            T: a generation either text, image, audio, or video.
        """
        ...

    # TODO: Rethink the output type, e.g., generator of bytes.
    def batched_generate(
        self,
        prompts: List[str],
        generation_config: Dict,
    ) -> List[T]:
        """Generates using each prompt in the list of `prompts`.

        Args:
            prompts (List[str]): List of prompts to generate completions for.
            generation_config (Dict): Dictionary containing the generation parameters.

        Returns:
            List[T]: a list of generations
        """
        completions, responses = [], []
        with ThreadPoolExecutor(
            max_workers=min(self.model_config.threads, len(prompts))
        ) as thread_pool:
            for prompt in prompts:
                responses.append(
                    thread_pool.submit(
                        self.sample_generate, prompt, generation_config
                    )
                )
            # Wait completions
            completions = [response.result() for response in tqdm(responses)]

        return completions
