from typing import List

import spacy
from tqdm import tqdm
from math import ceil
from random import uniform, choice, sample
from ..common import InvalidSpacyModel, get_logger
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
    spacy_model = SPACY_MODEL_MAPPING.get(language, None)
    if spacy_model is None:
        raise InvalidSpacyModel(language)

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


def mask_spans(text: str) -> str:
    ...


def format_mask_token(idx: int, mask_token: str, add_period: bool = False) -> str:
    period = "." if add_period else ""
    return f"{mask_token}-{idx}{period}"


def mask_sentences(
    texts: List[str], language: str, percentage_range: List[float], mask_token: str
) -> List[str]:
    """Mask a random percentage of sentences in the texts provided.

    Args:
        texts (List[str]): list of texts to mask.
        language (str): language for the spacy pipeline.
        percentage_range (List[float]): min and max percentage to mask.
        mask_token (str): token to replace masked sentences.

    Returns:
        List[str]: list of texts with masked sentences.
    """
    docs = spacy_pipeline(texts, language)
    masked_texts = []
    for doc in docs:
        sentences = [sent.text for sent in doc.sents]
        # In case of 1 sentences, avoid masking it,
        # just add a mask sentence to right or left randomly.
        if len(sentences) == 1:
            mask_position = choice([0, 1])
            fmt_masked_token = format_mask_token(
                0,
                mask_token=mask_token,
                add_period=True,
            )
            sentences.insert(mask_position, fmt_masked_token)
            masked_texts.append(" ".join(sentences).strip())
        else:
            percentage_masks = uniform(*percentage_range)
            # Min to avoid masking all the sentences.
            sents_to_mask = min(
                len(sentences) - 1, ceil(percentage_masks * len(sentences))
            )
            positions = sorted(sample(range(len(sentences)), sents_to_mask))
            for mask_idx, sent_idx in enumerate(positions):
                sentences[sent_idx] = format_mask_token(
                    mask_idx,
                    mask_token=mask_token,
                    add_period=True,
                )
            masked_texts.append(" ".join(sentences).strip())
    return masked_texts
