# flake8: noqa
from typing import Mapping, Type

from ..config import Config
from ..types import TaskType
from .attribution import AttributionDatasetGenerator
from .base import DatasetGenerator
from .boundary import BoundaryDatasetGenerator
from .detection import DetectionDatasetGenerator

GENERATORS: Mapping[TaskType, Type[DatasetGenerator]] = {
    TaskType.DETECTION: DetectionDatasetGenerator,
    TaskType.ATTRIBUTION: AttributionDatasetGenerator,
    TaskType.BOUNDARY: BoundaryDatasetGenerator,
}


def get_generator(config: Config) -> DatasetGenerator:
    """
    Gets a dataset generator from the pool.

    Args:
        config (Config): a config.

    Returns:
        DatasetGenerator: a dataset generator from the pool.
    """
    return GENERATORS[config.task_type](config)  # type: ignore


__all__ = [str(cls) for cls in GENERATORS.values()]
