from typing import Dict, List

from datasets import Dataset

from ..config import InputConfig
from ..types import TaskType
from .base import Extractor


class Dummy(Extractor):
    """
    Dummy extractor that fills the prompt template with empty texts.

    This extractor needs one template placeholder named {dummy}.

    This extractor does not need specific arguments.
    """

    def __init__(self, input_config: InputConfig, task_type: TaskType):
        super().__init__(input_config, task_type)

    def _extract(self, dataset: Dataset) -> Dict[str, List[str]]:
        return {
            "dummy": [
                ""
                for _ in range(
                    len(dataset[self.input_config.dataset_text_column])
                )
            ]
        }
