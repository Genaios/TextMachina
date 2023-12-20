from importlib import import_module
from pathlib import Path
from typing import Dict, List, Mapping

from datasets import Dataset

from ..common.exceptions import InvalidMetric, MissingMetricError
from ..types import TaskType
from .base import Metric

METRICS: Mapping[str, str] = {
    "mauve": "MAUVEMetric",
    "repetition_diversity": "RepetitionDiversityMetric",
    "simple_model": "SimpleModelMetric",
    "perplexity": "PerplexityMetric",
}


def run_metrics(
    dataset: Dataset,
    task_type: TaskType,
    save_dir: Path,
    metrics: List[str],
    metric_args: Dict[str, Dict],
) -> None:
    """
    Runs a list of metrics against a dataset.

    Args:
        dataset (Dataset): the dataset where metrics will be applied.
        task_type (TaskType): the type of task.
        save_dir (Path): the path where metric outputs will be saved.
        metrics: (List[str]): the list of metrics to run.
        metric_args: (Dict): the arguments of each metric
    """
    for name in metrics:
        metric = get_metric(task_type, name)
        metric.run(dataset, save_dir, **metric_args[name])


def get_metric(task_type: TaskType, name: str) -> Metric:
    """
    Gets a metric from the pool.

    Args:
        task_type (TaskType): the type of task.
        name (str): the metric name.
    Returns:
        Metric: the selected metric from the pool.

    """
    metric_cls_name = METRICS.get(name, None)
    if metric_cls_name is None:
        raise InvalidMetric(name)

    try:
        metric_class = getattr(
            import_module(f".{name}", __name__),
            metric_cls_name,
        )
    except (ModuleNotFoundError, ImportError):
        raise MissingMetricError(metric=name)
    return metric_class(task_type, name)


__all__ = list(METRICS.values())
