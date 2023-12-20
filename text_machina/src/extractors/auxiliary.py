import re
from typing import Dict, List

from datasets import Dataset

from ..config import InputConfig
from ..types import TaskType
from .base import Extractor


class Auxiliary(Extractor):
    def __init__(self, input_config: InputConfig, task_type: TaskType):
        super().__init__(input_config, task_type)

    def _extract(self, dataset: Dataset) -> Dict[str, List[str]]:
        regex = r"\{(\w+)\}"
        columns = re.findall(regex, self.input_config.template)
        return {
            column: dataset[column]
            for column in columns
            if column in dataset.features
        }
