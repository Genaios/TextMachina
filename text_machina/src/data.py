import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)

from .common.logging import get_logger
from .config import Config, InputConfig
from .extractors import get_extractor
from .models.types import GENERATION_ERROR
from .tokenizers import get_tokenizer
from .types import Prompt, PromptedDataset, TaskType

_logger = get_logger(__name__)


class PromptedDatasetBuilder:
    """
    Class to manage all the prompting steps required before generating MGT.
    """

    def __init__(self, config: Config):
        self.config = config
        self.prompt = self.get_prompt()
        self.extractor = get_extractor(
            self.prompt.extractor, self.config.input, self.config.task_type
        )

    def build(self) -> PromptedDataset:
        """
        Prepares prefixes based on input formats for a particular
        domain, model and dataset.

        Returns:
            PromptedDataset: a dataset with prompted and human texts.
        """
        # load and prepare dataset
        dataset = load_dataset_from_config(self.config.input)

        # sample human texts
        human_texts, dataset = self.sampling(dataset)

        # compute prompt inputs and prepare human texts
        prompt_inputs = self.extractor.extract(dataset)
        human_texts = self.extractor.prepare_human(human_texts)

        # truncate the prompt inputs and format the prompts
        prompt_inputs = self.truncate_inputs(prompt_inputs)
        inputs = format_prompt(self.prompt.template, prompt_inputs)

        return PromptedDataset(prompted_texts=inputs, human_texts=human_texts)

    def sampling(self, dataset: Dataset) -> Tuple[List[str], Dataset]:
        """
        Sample human texts and texts to be used for generating MGT.
        The same amount is sampled in both cases.

        This method allows to randomly sample human texts, or use
        the same ones than those that will be used to generate MGT.

        Args:
            dataset (Dataset): a dataset.
        Returns:
            Tuple[List[str], Dataset]: tuple of texts. human texts and
                texts to be used to generate MGT.
        """
        dataset = dataset.shuffle()
        select_range = range(min(self.config.input.quantity, len(dataset)))

        # Disable random_sample_human automatically for boundary tasks
        if self.config.task_type == TaskType.BOUNDARY:
            _logger.info(
                "Automatically disabling `random_sample_human`"
                f"for the {TaskType.BOUNDARY.value} task."
            )
            self.config.input.random_sample_human = False

        if self.config.input.random_sample_human:
            human_texts = dataset.select(select_range)[
                self.config.input.dataset_text_column
            ]
            dataset = dataset.shuffle()
            dataset = dataset.select(select_range)
        else:
            dataset = dataset.select(select_range)
            human_texts = dataset[self.config.input.dataset_text_column]
        return human_texts, dataset

    def get_prompt(self) -> Prompt:
        """
        Returns the input format to be used as input for
        the text generation models

        Returns:
            Prompt: a prompt with template and extractor.
        """
        prompt = Prompt(
            template=self.config.input.template,
            extractor=self.config.input.extractor,
        )
        _logger.info(f"Prompt prepared: {prompt}")
        return prompt

    def truncate_inputs(
        self, prompt_inputs: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Truncates prompt inputs extracted with the extractors.

        Args:
            prompt_inputs (Dict[str, List[str]]): prompt inputs.

        Returns:
            Dict[str, List[str]]: truncated prompt inputs.
        """

        max_input_tokens = self.config.input.max_input_tokens
        tokenizer = get_tokenizer(
            provider=self.config.model.provider,
            model_name=self.config.model.model_name,
        )

        prompt_inputs = tokenizer.distributed_truncate(
            prompt_inputs, max_input_tokens
        )

        _logger.info(f"Truncated prompt inputs to {max_input_tokens} tokens.")

        return prompt_inputs


def format_prompt(
    template: str, prompt_inputs: Dict[str, List[str]]
) -> List[str]:
    """
    Formats a prompt template with the prompt inputs.

    Example:
        template: "Write a text using this entities: {entities}.\n Text:"
        prompt_inputs: {"entities": ["Jose, Areg", "Marc, Angelo"]}
        output: ["Write a text using this entities: Jose, Areg.\n Text:",
                 "Write a text using this entities: Marc, Angelo.\n Text:"

    Args:
        template (str): the template to be formatted.
        prompt_inputs (Dict[str, List[str]]): prompt inputs from the extractors

    Returns:
        List[str]: formatted templates.
    """
    result = []
    input_keys = list(prompt_inputs.keys())
    for i in range(len(prompt_inputs[input_keys[0]])):
        args = {key: prompt_inputs[key][i] for key in input_keys}
        result.append(template.format(**args))
    return result


def serialize_dataset(
    dataset: Dataset,
    config: Config,
    path: Path,
    run_name: str,
) -> Path:
    """
    Saves a dataset with its config as an additional column.

    Args:
        dataset (Dataset): a dataset.
        config (Config): configuration used.
        path (Path): path where to save the generated dataset.
        run_name (str): name of this run.

    Returns:
        Path: folder where the dataset was saved.
    """
    save_path = get_save_path(config, path, run_name)

    save_path.mkdir(parents=True, exist_ok=True)

    dataset.save_to_disk(save_path.as_posix())
    _logger.info(f"The dataset has been saved in {save_path}")

    return save_path


def concatenate(paths: List[Path], save_path: Path) -> Dataset:
    """
    Concatenates and saves a list of datasets.

    Args:
        paths (List[Paths]): list with the datasets to be concatenated.
        save_path (Path): path where to save the merged dataset.

    Returns:
        Dataset: the merged dataset.
    """
    save_path.mkdir(parents=True, exist_ok=True)
    dataset = load_from_disk(paths[0].as_posix())

    for path in paths[1:]:
        dataset = concatenate_datasets(
            [dataset, load_from_disk(path.as_posix())]
        )

    return dataset


def load_dataset_from_config(config: InputConfig) -> Dataset:
    """
    Loads a dataset from disk or hub.

    Args:
        config (InputConfig): an input config.
    Returns:
        Dataset: a dataset.
    """
    try:
        dataset = load_from_disk(config.dataset)
        if "split" in config.dataset_params:
            dataset = dataset[config.dataset_params["split"]]
    except FileNotFoundError:
        dataset = load_dataset(config.dataset, **config.dataset_params)

    if isinstance(dataset, DatasetDict):
        split = list(dataset.keys())[0]
        _logger.warn(
            f"Picking the {split} split, since it was not"
            "specified in the config file."
        )
        dataset = dataset[split]

    return dataset


def get_save_path(
    config: Config, save_dir: Path, run_name: str, check_exists: bool = False
) -> Path:
    """
    Constructs the path to save a dataset.

    Args:
        config (Config): config of this run.
        save_dir (Path): root of the save path.
        run_name (str): name of this run.
        check_exists (bool): ...

    Returns:
        Path: path to save a dataset.

    """
    config_as_string = json.dumps(
        config.model_dump(), sort_keys=True, default=str
    ).encode()
    prefix = hashlib.sha256(config_as_string).hexdigest()

    parent = save_dir / run_name
    save_path = parent / prefix

    # check if path with prefix exists
    if check_exists:
        return get_path_from_substring(parent, prefix)  # type: ignore

    return save_path


def get_path_from_substring(path: Path, substring: str) -> Optional[Path]:
    """
    Checks whether a folder name within `path` includes `substring`

    Args:
        path (Path): path where searching for folders.
        substring (str): substring to find in the names.

    Returns:
        Optional[Path]: a path of a folder named *`substring`* or None.
    """
    for p in path.glob("*"):
        if substring in p.name:
            return p

    return None


def domain_model_counts(dataset: Dataset) -> pd.DataFrame:
    """
    Computes counts for (domain, model) pairs, e.g:

        model    bloom-560m  gpt2  human  total
        domain
        reviews          10    10     20     40
        tweets           10    10     20     40
        total            20    20     40     80

    Args:
        dataset (Dataset): the dataset used to compute counts.

    Returns:
        pd.DataFrame: the (domain, model) counts.
    """
    df = dataset.to_pandas()

    # We're interested in the number of available examples,
    # so we ignore the errors in the generation process.
    df = df[~df["text"].str.contains(GENERATION_ERROR)]

    counts = (
        df.groupby(["model", "domain"]).size().rename("count").reset_index()
    )

    counts = counts.pivot(columns="model", index="domain", values="count")

    counts.loc["total"] = counts.sum(axis=0)
    counts["total"] = counts.sum(axis=1)

    return counts


def errors_per_model(dataset: Dataset) -> pd.DataFrame:
    """
    Computes error counts per model.

    Args:
        dataset (Dataset): the dataset used to compute counts.

    Returns:
        pd.DataFrame: the error counts per model.
    """
    df = dataset.to_pandas()

    df["errors"] = df["text"].str.contains(GENERATION_ERROR)
    df["texts"] = ~df["errors"]

    counts = df[["model", "texts", "errors"]].groupby("model").sum()
    counts.loc["total"] = counts.sum(axis=0)

    return counts
