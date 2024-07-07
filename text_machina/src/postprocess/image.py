from functools import wraps
from typing import Callable, Dict, List, Tuple

from datasets import Dataset, disable_caching
from PIL.Image import Image

from ..common.logging import get_logger

_logger = get_logger(__name__)

disable_caching()


# TODO: Unify this for all postprocessings by allowing to pass `input_columns`
def batched_map(
    f: Callable[[List[str]], Dict[str, List[str]]]
) -> Callable[[Dataset], Dataset]:
    """
    Runs a function `f` on a dataset with batched mapping.

    Args:
        f (Callable[[List[str]], Dict[str, List[str]]]): the function.

    Returns:
        Callable[[Dataset], Dataset]: the modified function.
    """

    @wraps(f)
    def with_batched_mapping(dataset: Dataset) -> Dataset:
        """Applies batched map to a function, shows name in progress bar"""
        desc = " ".join(f.__name__.split("_")).capitalize()
        dataset = dataset.map(
            f,
            input_columns=["image"],
            batched=True,
            load_from_cache_file=False,
            desc=desc,
        )
        return dataset

    return with_batched_mapping


@batched_map
def resize(
    images: List[Image], size: Tuple[int, int] = (512, 512)
) -> Dict[str, List[Image]]:
    """
    Resizes all the images to a fixed size.
    Args:
        TODO
    Returns:
        TODO
    """
    resized = []
    for image in images:
        resized.append(image.resize(size))
    return {"image": resized}


def remove_generation_errors(dataset: Dataset) -> Dataset:
    """
    Removes generation errors. `None` in case of images.

    Args:
        dataset (Dataset): the dataset to filter.
    Returns:
        Dataset: a filtered dataset with no error annotations.
    """
    return dataset.filter(lambda x: x["image"] is not None)


def postprocess(dataset: Dataset) -> Dataset:
    """
    Postprocesses a dataset.

    Args:
        dataset (Dataset): the dataset to postprocess.
    Returns:
        Dataset: the postprocessed dataset.
    """

    single_text_actions = [resize]

    full_dataset_actions = [
        remove_generation_errors,
    ]

    actions = full_dataset_actions + single_text_actions

    for action in actions:
        dataset = action(dataset)  # type: ignore

    return dataset
