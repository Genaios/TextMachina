# flake8: noqa
from typing import Mapping, Type

from ..config import InputConfig
from ..types import TaskType
from .auxiliary import Auxiliary
from .base import Extractor
from .combined import Combined
from .dummy import Dummy
from .entity_list import EntityList
from .noun_list import NounList
from .sentence_prefix import SentencePrefix
from .word_prefix import WordPrefix

EXTRACTORS: Mapping[str, Type[Extractor]] = {
    "auxiliary": Auxiliary,
    "combined": Combined,
    "dummy": Dummy,
    "entity_list": EntityList,
    "noun_list": NounList,
    "sentence_prefix": SentencePrefix,
    "word_prefix": WordPrefix,
}


def get_extractor(
    extractor: str, input_config: InputConfig, task_type: TaskType
) -> Extractor:
    """
    Gets an extractor from the extractors pool.

    Args:
        extractor (str): extractor name.
        input_config (InputConfig): an input config.

    Returns:
        Extractor: an extractor from the pool.
    """
    return EXTRACTORS[extractor](input_config, task_type)


__all__ = [str(cls) for cls in EXTRACTORS.values()]
