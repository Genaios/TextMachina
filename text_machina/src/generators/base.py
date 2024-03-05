from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from datasets import Dataset

from ..common.logging import get_logger
from ..config import Config
from ..constrainers import get_length_constrainer
from ..data import PromptedDatasetBuilder
from ..models import get_model

_logger = get_logger(__name__)


class DatasetGenerator(ABC):
    """
    Base class for dataset generators.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = get_model(self.config.model)
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

    def _generate(self) -> Tuple[List[str], Dict]:
        """
        Generates a dataset based on the provided config.

        Returns:
            Tuple[List[str], Dict]: a tuple of the generated texts and
                additional arguments to use for dataset packing.
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

    @abstractmethod
    def _pack(self, generations: List[str], **kwargs) -> Dataset:
        """
        Builds a dataset by packing the texts accordingly to the task.

        Args:
            generations (List[str]): the generated texts.
            **kwargs: additional keyword arguments.

        Returns:
            Dataset: the final labeled dataset.
        """
        ...

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
        dataset = dataset.add_column(
            "language",
            [str(self.config.input.language)] * len(dataset),
        )
        return dataset
