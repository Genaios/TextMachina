from typing import List

from datasets import Dataset, concatenate_datasets

from ..common.exceptions import DatasetGenerationError
from ..config import Config
from ..types import DetectionLabels, Placeholders
from .base import SpanDatasetGenerator


class SpanDetectionDatasetGenerator(SpanDatasetGenerator):
    """
    Dataset generator for the span detection task type.

    Implements `_pack` by correctly labeling the dataset for detection.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config=config)

    def _pack(self, generations: List[str], **kwargs) -> Dataset:
        """
        Combines and labels the generated and human texts.

        Args:
            generations (List[str]): list of generated texts.
            prompted_dataset (PromptedDataset): dataset with prompts and human texts.
            kwargs: additional keyword arguments.

        Returns:
            Dataset: a dataset including all the texts.
        """
        prompted_dataset = kwargs.get("prompted_dataset", None)
        if prompted_dataset is None:
            raise DatasetGenerationError(f"prompted_dataset not found: {self}")
        prompt_inputs = prompted_dataset.prompt_inputs.values()
        print(prompt_inputs)
        print(generations)
        exit()
        model_name = self.config.model.model_name
        domain = self.config.input.domain
        extractor = self.config.input.extractor

        generated_dataset = Dataset.from_list(
            [
                {
                    "prompt": prompt,
                    "text": text,
                    "label": DetectionLabels.GENERATED.value,
                    "model": model_name,
                    "domain": domain,
                    "extractor": extractor,
                }
                for prompt, text in zip(prompted_dataset.prompted_texts, generations)
            ]
        )

        human_dataset = Dataset.from_list(
            [
                {
                    "prompt": Placeholders.NO_PROMPT.value,
                    "text": text,
                    "label": DetectionLabels.HUMAN.value,
                    "model": DetectionLabels.HUMAN.value,
                    "domain": domain,
                    "extractor": Placeholders.NO_EXTRACTOR.value,
                }
                for text in prompted_dataset.human_texts
            ]
        )

        dataset = concatenate_datasets([human_dataset, generated_dataset])
        dataset = dataset.shuffle()

        return dataset
