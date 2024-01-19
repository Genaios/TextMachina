# flake8: noqa
from typing import Mapping, Type

from datasets import Dataset

from ..types import TaskType
from .attribution import AttributionExplorer
from .base import Explorer
from .boundary import BoundaryExplorer
from .detection import DetectionExplorer
from .mixcase import MixCaseExplorer

EXPLORERS: Mapping[str, Type[Explorer]] = {
    TaskType.DETECTION: DetectionExplorer,
    TaskType.ATTRIBUTION: AttributionExplorer,
    TaskType.BOUNDARY: BoundaryExplorer,
    TaskType.MIXCASE: MixCaseExplorer,
}


def get_explorer(task_type: TaskType, dataset: Dataset) -> Explorer:
    """
    Gets an explorer from the pool.

    Args:
        task_type (TaskType): the task type.
        dataset (Dataset): a dataset.

    Returns:
        Explorer: an explorer from the pool.
    """
    return EXPLORERS[task_type](dataset)


__all__ = [str(cls) for cls in EXPLORERS.values()]
