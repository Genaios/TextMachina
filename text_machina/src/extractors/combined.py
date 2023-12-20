import importlib
from functools import reduce
from typing import Dict, List

from datasets import Dataset

from ..config import InputConfig
from ..types import TaskType
from .base import Extractor


class Combined(Extractor):
    def __init__(self, input_config: InputConfig, task_type: TaskType):
        super().__init__(input_config, task_type)
        self.extractors = []
        for extractor_name in self.input_config.extractors_list:
            module, class_ = extractor_name.rsplit(".", 1)
            extractor = getattr(
                importlib.import_module(
                    "." + module, package=__name__.rsplit(".", 1)[0]
                ),
                class_,
            )(self.input_config, task_type)
            self.extractors.append(extractor)

    def _extract(self, dataset: Dataset) -> Dict[str, List[str]]:
        extractor_outputs = [
            extractor._extract(dataset) for extractor in self.extractors
        ]
        return reduce(lambda x, y: dict(x, **y), extractor_outputs)
