import re
from typing import Dict, List

from datasets import Dataset

from ..config import InputConfig
from ..types import TaskType
from .base import Extractor
from .utils import spacy_pipeline


class GapSpan(Extractor):
    def __init__(self, input_config: InputConfig, task_type: TaskType):
        super().__init__(input_config, task_type)
        self.args = self.input_config.extractor_args.get("gap_span", {})

    def prepare_human(self, human_texts: List[str]) -> List[str]:
        return human_texts

    def _extract(self, dataset: Dataset) -> Dict[str, List[str]]:
        column = re.findall(r"\{(\w+)\}", self.input_config.template)[0]

        return {self.placeholder: output_texts}
