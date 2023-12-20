from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from datasets import Dataset

from ..common.logging import get_logger
from ..config import Config
from ..constrainers import get_length_constrainer
from ..data import PromptedDatasetBuilder
from ..models import get_model
from ..types import PromptedDataset

_logger = get_logger(__name__)


class DatasetGenerator(ABC):
    """
    Base class to manage the text generation process.

    Should be implemented for each `TaskType`.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = get_model(config.model)
        self.prompter = PromptedDatasetBuilder(self.config)

    def generate(self) -> Dataset:
        """
        Generates a labeled dataset based on the provided config.

        Returns:
            Dataset: the dataset
        """
        generations, kwargs = self._generate()
        dataset = self._pack(generations, **kwargs)
        dataset = self.add_config_info(dataset)
        return dataset

    def add_config_info(self, dataset: Dataset) -> Dataset:
        """
        Adds config information to the dataset.

        Args:
            dataset (Dataset): the dataset to add config information to.
        Returns:
            Dataset: the dataset with config information added.
        """
        dataset = dataset.add_column(
            "config_path",
            [str(self.config.path)] * len(dataset),
        )
        return dataset

    @abstractmethod
    def _generate(self) -> Tuple[List[str], Dict]:
        """
        Generates a dataset based on the provided config.

        Returns:
            Tuple[List[str], Dict]: a tuple of the generated texts and
                additional arguments to use for dataset packing.
        """
        ...

    @abstractmethod
    def _pack(self, generations: List[str], **kwargs) -> Dataset:
        """
        Labels, combines, generally packs the generated texts.

        Args:
            generations (List[str]): the generated texts.
            *args: additional arguments.
            **kwargs: additional keyword arguments.

        Returns:
            Dataset: the final labeled dataset.
        """
        ...


class ClassificationDatasetGenerator(DatasetGenerator):
    """
    Dataset generator for classification tasks such as detection or attribution.

    Implements `_generate` specifically for classification tasks.
    Note that `_pack` still needs to be implemented.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config=config)

    def _generate(self) -> Tuple[List[str], Dict[str, PromptedDataset]]:
        """
        Carries out generation process:
        - Builds a prompt for each human sample in a given dataset
        - Constrains generation length based on human length distribution
        - Runs model inference.
        """
        # prepare inputs
        prompted_dataset = self.prompter.build()
        _logger.info(
            f"This is how one input looks like: {prompted_dataset.prompted_texts[0]}"
        )

        # instantiate length constrainer
        length_constrainer = get_length_constrainer(
            texts=prompted_dataset.human_texts,
            model_name=self.config.model.model_name,
            provider=self.config.model.provider,
        )

        # constrain generation config
        generation_config = length_constrainer.constrain(self.config.generation)

        _logger.info(
            f"Generating completions for with args:\n"
            f"Model: {self.config.model}.\n"
            f"Args: {generation_config}"
        )

        # run generator
        generations = self.model.generate_completions(
            prompts=prompted_dataset.prompted_texts,
            generation_config=generation_config,
        )

        return generations, {"prompted_dataset": prompted_dataset}
