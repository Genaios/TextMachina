from abc import ABC, abstractmethod
from typing import Dict, List

from datasets import Dataset

from ..config import InputConfig
from ..types import TaskType
from .utils import clean_inputs


class Extractor(ABC):
    def __init__(self, input_config: InputConfig, task_type: TaskType):
        self.input_config = input_config
        self.task_type = task_type

    @abstractmethod
    def _extract(self, dataset: Dataset) -> Dict[str, List[str]]:
        """
        Returns the prompt inputs for each sample in a dataset.
        This method must be overridden in each new extractor.

        Example:
            input = Dataset({"text": ["hi Jose", "hi Areg"], "label": [0, 1]})
            output = {"entities": ["Jose", "Areg"], "interject": ["hi", "hi"]}

        Args:
            dataset (Dataset): A dataset to extract inputs from.

        Returns:
            Dict[str, List[str]]: A dictionary mapping each template
                                  key to a list of prompt inputs
                                  (one input per template key and example).
        """
        ...

    def prepare_human(self, human_texts: List[str]) -> List[str]:
        """
        Prepares the human texts. Some extractors could need to modify
        human texts according to the extractions, e.g., remove prefixes
        from texts to ensure that generations and human texts are
        continuations of the same prefix.

        Args:
            human_texts (List[str]): list of human texts.

        Returns:
            List[str]: prepared human texts.
        """
        return human_texts

    def extract(self, dataset: Dataset) -> Dict[str, List[str]]:
        """
        Calls _extract and cleans the extracted inputs.

        Args:
            dataset (Dataset): A dataset to extract inputs from.

        Returns:
            Dict[str, List[str]]: A dictionary mapping each template
                                  key to a list of prompt inputs
                                  (one input per template key and example).
        """
        prompt_inputs = self._extract(dataset)
        prompt_inputs = {
            column: clean_inputs(prompt_inputs[column])
            for column in prompt_inputs
        }
        return prompt_inputs
