from typing import List

from datasets import Dataset

from ..common.exceptions import DatasetGenerationError
from ..config import Config
from .base import DatasetGenerator


class AttributionDatasetGenerator(DatasetGenerator):
    """
    Dataset generator for the attribution task type.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config=config)

    def _pack(self, generations: List[str], **kwargs) -> Dataset:
        """
        Combines and labels the generated and human texts.

        Args:
            generations (List[str]): list of generated texts.
            kwargs: additional keyword arguments.

        Returns:
            Dataset: a dataset including all the texts.
        """

        prompted_dataset = kwargs.get("prompted_dataset", None)
        if prompted_dataset is None:
            raise DatasetGenerationError(f"prompted_dataset not found: {self}")

        model_name = self.config.model.model_name
        domain = self.config.input.domain
        extractor = self.config.input.extractor

        dataset = Dataset.from_list(
            [
                {
                    "prompt": prompt,
                    "text": text,
                    "label": model_name,
                    "model": model_name,
                    "domain": domain,
                    "extractor": extractor,
                }
                for prompt, text in zip(
                    prompted_dataset.prompted_texts, generations
                )
            ]
        )

        dataset = dataset.shuffle()

        return dataset
