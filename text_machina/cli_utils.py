from logging import FileHandler
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import petname
from datasets import Dataset

from .src.common.logging import get_logger
from .src.common.utils import get_cache_path
from .src.config import Config, parse_metrics_config
from .src.data import (
    concatenate,
    domain_model_counts,
    errors_per_model,
    get_save_path,
    serialize_dataset,
)
from .src.generators import get_generator
from .src.interactive import step as _step
from .src.metrics import run_metrics as _run_metrics
from .src.models.types import GENERATION_ERROR
from .src.postprocessing import filter_by_language, postprocess
from .src.types import TaskType

_logger = get_logger(__name__)


def generate_from_config(
    config: Config,
    save_dir: Path,
    run_name: str,
) -> Path:
    """
    Generates a dataset using `TextGeneration` parameterized
    by `config` and saves it.

    Args:
        config (Config): a configuration.
        save_dir (Path): root dir where to save the generated dataset.
        run_name (str): name of this run.

    Returns:
        Path: path where the generated dataset was saved.
    """
    generator = get_generator(config)
    dataset = generator.generate()

    errors = count_errors(dataset)
    _logger.info(f"{errors} errors found in the generated dataset.")

    dataset = filter_by_language(dataset, config.input.language)

    output_path = serialize_dataset(dataset, config, save_dir, run_name)

    return output_path


def _generate(
    config_path: Path,
    save_dir: Path,
    run_name: str,
    task_type: TaskType,
) -> None:
    """
    Runs the generation pipeline in an end-to-end manner.

    Args:
        config_path (Path): path containing YAML config files.
        save_dir (Path): root dir where to save the generated dataset.
        run_name (str): name of this run.
        task_type (TaskType): the type of task.
    """
    configs = Config.load_configs(config_path, task_type)

    _, statistics = generate_dataset(configs, save_dir, run_name)

    statistics_dir = save_dir / "statistics"
    statistics_dir.mkdir(parents=True, exist_ok=True)
    for name, df in statistics.items():
        df.to_markdown(statistics_dir / f"{name}.md")
        df.to_json(statistics_dir / f"{name}.json")


def generate_dataset(
    configs: List[Config],
    save_dir: Path,
    run_name: str,
) -> Tuple[Dataset, Dict[str, pd.DataFrame]]:
    """
    Generates a dataset given a list of configs.
    Only generates a dataset for a config if it hasn't
    been already generated for this `run_name`.

    Computes statistics for the generated dataset.

    Args:
        configs (List[Config]): list of configs to use for generation.
        save_dir (Path): root dir where to save the generated dataset.
        run_name (str): name of this run.
    Returns:
        Tuple[Dataset, Dict[str, pd.DataFrame]]: a tuple (dataset, statistics dict)
    """
    cache_path = get_cache_path()
    paths = []

    for config in configs:
        path = get_save_path(config, cache_path, run_name, check_exists=True)
        if not path:
            path = generate_from_config(config, cache_path, run_name)

        paths.append(path)

    dataset = concatenate(paths, save_dir)
    statistics = compute_statistics(dataset)
    errors = count_errors(dataset)

    dataset = postprocess(dataset, configs[0].task_type)

    dataset.save_to_disk(save_dir)

    _logger.info(
        f"A total of {errors} errors have been found in the generation process."
    )
    _logger.info(f"Your dataset has been generated at {str(save_dir)}")

    return dataset, statistics


def compute_statistics(dataset: Dataset) -> Dict[str, pd.DataFrame]:
    """
    Computes a set of statistics of a generated dataset.

    Args:
        dataset (Dataset): the dataset of which statistics are computed.
    Returns:
        Dict[str, pd.DataFrame]: the statistics.
    """
    domain_model = domain_model_counts(dataset)
    errors = errors_per_model(dataset)

    return {"domain_model": domain_model, "errors": errors}


def generate_run_name() -> str:
    """
    Generates a name for a run.

    Returns:
        str: name of the run.
    """
    return petname.generate()


def log_final_message(run_name: str) -> None:
    """
    Logs the last logging message of TextMachina.

    Args:
        run_name (str): name of the run.
    """
    _logger.info(
        f"This run has been registered with name: '{run_name}'."
        " If the run terminated due to errors, you can use this"
        " name to continue from the where the process left off."
    )
    file_handler = [h for h in _logger.handlers if isinstance(h, FileHandler)][
        0
    ]
    _logger.info(f"Logfile saved in {file_handler.baseFilename}")


def count_errors(dataset: Dataset) -> int:
    """
    Counts the number of generation errors in the dataset.
    A generation error is identified as a text being `GENERATION_ERROR`.

    Args:
        dataset (Dataset): a dataset.

    Returns:
        int: number of errors in the dataset.
    """
    error_count = sum(
        [x in x for x in dataset["text"] if x == GENERATION_ERROR]
    )
    return error_count


def _explore(
    config_path: Path,
    metrics_path: Optional[Path],
    save_dir: Path,
    run_name: str,
    task_type: TaskType,
    step: bool,
    max_generations: int,
) -> None:
    """
    Carries out the exploration steps:
    - create a small dataset
    - generate a set of metrics based on the task type
    - step through the dataset

    Args:
        config_path (Path): path containing YAML config files for generation.
        metrics_path (Optional[Path]): path to YAML config file of metrics.
        save_dir (Path): root dir where to save the generated dataset.
        run_name (str): name of this run.
        task_type (TaskType): the type of task to be explored.
        step (bool): whether to step through the dataset.
        max_generations (int): the maximum number of texts to generate
            for each config. Ignored if the dataset has already been generated.
    """

    configs = Config.load_configs(
        config_path, task_type=task_type, max_generations=max_generations
    )

    dataset, _ = generate_dataset(configs, save_dir, run_name)

    if metrics_path:
        metrics, metric_args = parse_metrics_config(metrics_path)
        # Generate metrics
        if metrics:
            if metric_args is None:
                metric_args = {}
            _run_metrics(dataset, task_type, save_dir, metrics, metric_args)

    # Show examples in console
    if step:
        _step(dataset, task_type)
