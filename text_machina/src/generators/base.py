from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any

from datasets import Dataset

from ..common.logging import get_logger
from ..config import Config
from ..constrainers import get_length_constrainer
from ..data import PromptedDatasetBuilder
from ..models import get_model

_logger = get_logger(__name__)


class DatasetGenerator(ABC):
    """
    Base class to manage the text generation process.

    Should be implemented for each `TaskType`.
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


class SpanDatasetGenerator(DatasetGenerator):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.mask_token = self.config.input.extractor_args[self.config.input.extractor][
            "mask_token"
        ]

    def _merge_gaps(
        self, masked_text: str, generation: str
    ) -> Dict[str, List[Tuple[int, int]]]:
        ...


# dict_values([['Pritchard, a former England Under-21 international, has agreed a four-year deal at Carrow Road. MASK-0. MASK-1. He has made more than 400 career appearances for clubs including Exeter, Peterborough and Crawley and will fight for a first-team place alongside John Ruddy and Michael McGovern. Pritchard told the club website "I feel I need to go out and prove myself again in football. MASK-2. MASK-3. " Find all the latest football transfers on our dedicated page.']])
# ['{"MASK-0": "He is set to join the team in their upcoming season.", "MASK-1": "Pritchard brings a wealth of experience and talent to Norwich City", "MASK-2": "I believe Norwich is the perfect platform for me to do so.", "MASK-3": "I am excited for this new chapter in my career."}']
