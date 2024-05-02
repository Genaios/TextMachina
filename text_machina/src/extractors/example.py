import random
import re
from collections import defaultdict
from typing import Dict, List

from datasets import Dataset

from ..config import InputConfig
from ..types import TaskType
from .base import Extractor


class Example(Extractor):
    """
    Extractor that fills the prompt template with the text column
    of randomly sampled (w/o replacement) examples from the dataset.

    This extractor needs as many template placeholders named {example_i}
    as examples you want to include in the prompt.

    This extractor accepts the following arguments in the `extractor_args`
    field from the config:
        - all_random (bool): whether if the examples must be random
                             for each sample or not.
        - seed (int): the random seed.
    """

    def __init__(self, input_config: InputConfig, task_type: TaskType):
        super().__init__(input_config, task_type)
        self.args = self.input_config.extractor_args.get("example", {})
        self.all_random = self.args.get("all_random", True)
        self.seed = self.args.get("seed", 13)

    def _extract(self, dataset: Dataset) -> Dict[str, List[str]]:
        regex = r"\{(example_\d+)\}"
        example_placeholders = re.findall(regex, self.input_config.template)
        output = defaultdict(list)

        # Same examples for all the samples
        if not self.all_random:
            random_fixed = random.Random(self.seed)
            examples = dataset.select(
                random_fixed.sample(
                    range(len(dataset)), len(example_placeholders)
                )
            )[self.input_config.dataset_text_column]

            for idx, placeholder in enumerate(example_placeholders):
                output[placeholder] = [examples[idx]] * len(dataset)
        # Random examples for each sample
        else:
            for _ in range(len(dataset)):
                examples = dataset.select(
                    random.sample(
                        range(len(dataset)), len(example_placeholders)
                    )
                )[self.input_config.dataset_text_column]
                for idx, placeholder in enumerate(example_placeholders):
                    output[placeholder].append(examples[idx])

        return dict(output)
