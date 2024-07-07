from typing import List, Tuple

from datasets import Dataset, concatenate_datasets

from ..common.exceptions import DatasetGenerationError
from ..config import Config
from ..types import DetectionLabels, Modality, Placeholders
from .base import DatasetGenerator


class DetectionDatasetGenerator(DatasetGenerator):
    """
    Dataset generator for the detection task type.
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

        pack_fn = None
        if self.config.input.modality == Modality.TEXT:
            pack_fn = self._pack_text
        elif self.config.input.modality == Modality.IMAGE:
            pack_fn = self._pack_image

        generated_dataset, human_dataset = pack_fn(
            prompted_dataset, generations
        )

        dataset = concatenate_datasets([human_dataset, generated_dataset])
        dataset = dataset.shuffle()
        return dataset

    def _pack_text(
        self, prompted_dataset, generations
    ) -> Tuple[Dataset, Dataset]:
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
                for prompt, text in zip(
                    prompted_dataset.prompted_texts, generations
                )
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
        return generated_dataset, human_dataset

    def _pack_image(
        self, prompted_dataset, generations
    ) -> Tuple[Dataset, Dataset]:
        model_name = self.config.model.model_name
        domain = self.config.input.domain
        extractor = self.config.input.extractor
        generated_dataset = Dataset.from_list(
            [
                {
                    "prompt": prompt,
                    "text": text,
                    "image": image,
                    "label": DetectionLabels.GENERATED.value,
                    "model": model_name,
                    "domain": domain,
                    "extractor": extractor,
                }
                for prompt, text, image in zip(
                    prompted_dataset.prompted_texts,
                    prompted_dataset.human_texts,
                    generations,
                )
            ]
        )

        human_dataset = Dataset.from_list(
            [
                {
                    "prompt": prompt,
                    "text": text,
                    "image": image,
                    "label": DetectionLabels.HUMAN.value,
                    "model": DetectionLabels.HUMAN.value,
                    "domain": domain,
                    "extractor": Placeholders.NO_EXTRACTOR.value,
                }
                for prompt, text, image in zip(
                    prompted_dataset.prompted_texts,
                    prompted_dataset.human_texts,
                    prompted_dataset.human_images,
                )
            ]
        )
        return generated_dataset, human_dataset
