from typing import Dict, List, Set

import spacy
from datasets import Dataset

from ..config import InputConfig
from ..types import TaskType
from .base import Extractor
from .types import EXTRACTOR_OMITTED
from .utils import spacy_pipeline


def extract_nouns(processed_text: spacy.tokens.Doc) -> Set[str]:
    """
    Extracts noun chunks from a Spacy doc.

    Args:
        processed_text (Doc): Spacy doc.

    Returns:
        Set[str]: noun chunks in the doc.
    """
    # take noun chunks -> nouns -> roots -> disregard
    nouns = [x.text for x in processed_text.noun_chunks]
    if not nouns:
        nouns = [x.text for x in processed_text if x.pos_ == "NOUN"]
    if not nouns:
        nouns = [x.text for x in processed_text if x.dep_ == "ROOT"]
    else:
        nouns = [EXTRACTOR_OMITTED]  # to filter after
    return set(nouns)


class NounList(Extractor):
    """
    Extractor that fills the prompt template with noun-phrases
    extracted from a text column in the dataset.

    This extractor needs a template placeholder named {nouns}.

    This extractor does not need specific arguments.
    """

    def __init__(self, input_config: InputConfig, task_type: TaskType):
        super().__init__(input_config, task_type)

    def _extract(self, dataset: Dataset) -> Dict[str, List[str]]:
        processed_texts = spacy_pipeline(
            dataset[self.input_config.dataset_text_column],
            language=self.input_config.language,
            disable_pipes=[
                "ner",
                "senter",
                "attribute_ruler",
                "lemmatizer",
            ],
        )
        nouns = [", ".join(extract_nouns(text)) for text in processed_texts]
        return {"nouns": nouns}
