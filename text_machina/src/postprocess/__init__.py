# flake8: noqa
from datasets import Dataset

from ..types import Modality, TaskType
from .image import postprocess as postprocess_image
from .text import postprocess as postprocess_text


def postprocess(
    modality: str, dataset: Dataset, task_type: TaskType
) -> Dataset:
    if modality == Modality.IMAGE:
        return postprocess_image(dataset)
    elif modality == Modality.TEXT:
        return postprocess_text(dataset, task_type)
    else:
        raise ValueError(f"Modality {modality} not implemented yet.")
