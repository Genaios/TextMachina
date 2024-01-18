import re
from typing import Dict, List

from datasets import Dataset

from ..config import InputConfig
from ..types import TaskType
from .base import Extractor


class Auxiliary(Extractor):
    """
    Extractor that fills the prompt template with fields from
    a dataset.

    This extractor needs at least one template placeholder, named with
    the name of a field from the dataset, e.g., {summary}.

    This extractor does not need specific arguments.
    """

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
