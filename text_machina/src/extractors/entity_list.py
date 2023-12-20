from typing import Dict, List, Set

import spacy
from datasets import Dataset

from ..config import InputConfig
from ..types import TaskType
from .base import Extractor
from .types import EXTRACTOR_OMITTED
from .utils import spacy_pipeline


def extract_entities(processed_text: spacy.tokens.Doc) -> Set[str]:
    """
    Extracts entities from a Spacy doc.

    Args:
        processed_text (Doc): Spacy doc.

    Returns:
        Set[str]: named entities in the doc.
    """
    # take noun chunks -> nouns -> roots -> disregard
    entities = [x.text for x in processed_text.ents]
    if not entities:
        entities = [EXTRACTOR_OMITTED]  # to filter after
    return set(entities)


class EntityList(Extractor):
    def __init__(self, input_config: InputConfig, task_type: TaskType):
        super().__init__(input_config, task_type)

    def _extract(self, dataset: Dataset) -> Dict[str, List[str]]:
        processed_texts = spacy_pipeline(
            dataset[self.input_config.dataset_text_column],
            language=self.input_config.language,
            disable_pipes=[
                "senter",
                "attribute_ruler",
                "lemmatizer",
            ],
        )
        entities = [
            ", ".join(extract_entities(text)) for text in processed_texts
        ]
        return {"entities": entities}
