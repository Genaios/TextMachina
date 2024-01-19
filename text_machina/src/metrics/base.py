import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from datasets import Dataset

from ..common.logging import get_logger
from ..types import TaskType

_logger = get_logger(__name__)


class Metric(ABC):
    """
    Base class for metrics.
    """

    def __init__(self, task_type: TaskType, name: str) -> None:
        self.name = name
        self.task_type = task_type

    def run(self, dataset: Dataset, path: Path, **kwargs) -> None:
        """Calls _run, _save and _log to compute a metric, store and log its outputs.

        Args:
            dataset (Dataset): A labeled dataset on which to compute the metric.
            path (Path): A path where to save results.
            **kwargs: Additional arguments to pass to _run.
        """
        outputs = self._run(dataset, **kwargs)

        path = path / self.name
        path.mkdir(parents=True, exist_ok=True)

        self._save(outputs, path)
        self._log(outputs, _logger)

    @abstractmethod
    def _run(self, dataset: Dataset, **kwargs) -> Any:
        """Implements metric logic.

        This method must implement all computation of the metric, and
        must be overriden in each new metric implementation, returning
        any result (ideally in raw form) that will be used in _save or _log.

        Args:
            dataset (Dataset): A labeled dataset on which to compute the metric.
            **kwargs: Additional arguments to use for metric computation.

        Returns:
            Any: The outputs of computing the metric on the dataset.

        Raises:
            InvalidTaskTypeForMetric: if a metric is ran on a
                task type its not defined for.
            UnsupportedMetricParam: if a parameter in kwargs is unsupported.
                See exception docs for more information.
        """
        ...

    @abstractmethod
    def _save(self, outputs: Any, path: Path) -> None:
        """Implements saving logic.

        Any operations related to saving metric outputs should
        be carried out here, including their processing for saving.
        Must be overriden in each new metric implementation.

        Any saving operations must be carried out in `path` or within
        its children, without having to access parent directories.

        Args:
            outputs (Any): The outputs of computing the metric with _run.
            path (Path): the path where to save the
        """
        ...

    @abstractmethod
    def _log(self, outputs: Any, logger: logging.Logger) -> None:
        """Implements logging logic.

        Any operations related to logging metric outputs should
        be carried out here, including their processing.
        Must be overriden in each new metric implementation.

        All logging operations must be done through the `logger` object.

        Args:
            outputs (Any): The outputs of computing the metric with _run.
            logger: (logging.Logger): The logger object to use for logging.
        """
        ...
