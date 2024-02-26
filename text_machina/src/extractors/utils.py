from typing import List

import spacy
from tqdm import tqdm

from ..common import get_logger
from .types import SPACY_MODEL_MAPPING

_logger = get_logger(__name__)


def clean_inputs(texts: List[str]) -> List[str]:
    """
    Remove special symbols from the texts used as prompt inputs, to
    avoid breaking the format of classical kinds of prompts.

    Args:
        texts (List[str]): list of texts.
    Returns:
        List[str]: cleaned texts.
    """
    clean = []
    repl_map = {"\n": " ", "\t": " ", "\r": " ", "->": " ", ":": " "}
    for text in texts:
        for src, repl in repl_map.items():
            text = text.replace(src, repl)
        text = " ".join(text.split())
        text = text.strip()
        clean.append(text)
    return clean


def get_spacy_model(language: str) -> spacy.lang:
    """
    Gets or download a Spacy model.

    Args:
        language (str): language.

    Returns:
        spacy.lang: a Spacy model.
    """
    spacy_model = SPACY_MODEL_MAPPING.get(
        language, SPACY_MODEL_MAPPING["multilingual"]
    )

    try:
        nlp = spacy.load(spacy_model)
    except OSError:
        _logger.info(f"Downloading {spacy_model} from SpaCy.")
        spacy.cli.download(spacy_model)
        nlp = spacy.load(spacy_model)
    return nlp


def spacy_pipeline(
    texts: List[str],
    language: str,
    disable_pipes: List[str] = [],
    n_process: int = 4,
) -> List[spacy.tokens.Doc]:
    """
    Processes texts with spacy pipeline for entity extraction.

    Args:
        texts (List[str]): list of texts.
        language (str): language of the text.
        disable_pipes (List[str]): Spacy pipes to be disabled.
        n_process (int): number of processes.

    Returns:
        List[spacy.tokens.Doc]: list of Spacy docs.
    """
    nlp = get_spacy_model(language)

    processed_texts = list(
        tqdm(
            nlp.pipe(
                texts,
                n_process=n_process,
                disable=disable_pipes,
            ),
            total=len(texts),
            desc="Processing",
        )
    )

    return processed_texts
