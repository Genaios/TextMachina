import hashlib
import re
from functools import wraps
from typing import Callable, Dict, List

import fasttext
import ftfy
import numpy as np
import pandas as pd
import requests
from datasets import Dataset, disable_caching

from .common.logging import get_logger
from .common.utils import get_cache_path
from .models.types import GENERATION_ERROR
from .types import TaskType

_logger = get_logger(__name__)

disable_caching()

# Monkeypatch to disable fasttext warning:
# https://stackoverflow.com/questions/66353366/cant-suppress-fasttext-warning-load-model-does-not-return
fasttext.FastText.eprint = lambda x: None


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
            input_columns=["text"],
            batched=True,
            load_from_cache_file=False,
            desc=desc,
        )
        return dataset

    return with_batched_mapping


def remove_generation_errors(dataset: Dataset) -> Dataset:
    """
    Removes generation errors, i.e. texts marked with `GENERATION_ERROR`.

    Args:
        dataset (Dataset): the dataset to filter.
    Returns:
        Dataset: a filtered dataset with no error annotations.
    """
    return dataset.filter(lambda x: x["text"] != GENERATION_ERROR)


def truncate(
    dataset: Dataset,
    min_length: int = 5,
    min_tokens_to_truncate: int = 2,
    sampling_radius_size: float = 2.0,
) -> Dataset:
    """
    Truncates texts to remove token length bias per class in each domain.

    This is done by:
    1. Sampling the same number of texts per label in each domain
    2. Sorting them by token length
    3. Grouping them such that each group has one text per label
    4. Truncating the texts in the group to have the same length
    5. Truncating the remainder of 1. between mean +- 2*std
    6. Dropping texts with lengths < min_length

    Note that all the texts are truncated, this can be modified with
        min_tokens_to_truncate = 0.

    Args:
        dataset (Dataset): the dataset to truncate.
        min_length (int): the minimum (spacy) token length.
        min_tokens_to_truncate (int): the minimum (spacy) tokens to truncate.
        sampling_radius_size (float): the radius size for the sampling of
            token lengths for non-grouped texts.

    Returns:
        Dataset: the truncated dataset
    """
    from .extractors.utils import spacy_pipeline

    np.random.seed(0)

    df = dataset.to_pandas()

    # tokenize with spacy multilingual model
    df["tokenized"] = spacy_pipeline(
        df["text"],
        "multilingual",
    )
    # get tokenized length: we'll keep track of this and a
    # "difference" column to know how much to truncate
    # so then we only have to run the truncation itself once
    df["token_length"] = df["tokenized"].apply(len)

    to_truncate = []
    # set new token lengths (actual truncation happens later)
    for domain in df["domain"].unique():
        # sample enough data per label
        min_size = df[df["domain"] == domain].groupby("label").size().min()
        grouped = df[df["domain"] == domain].groupby("label").sample(min_size)

        # the remainder will be truncated based on mean +-std estimations
        remainder = df[~df.index.isin(grouped.index)].copy()
        remainder = remainder[remainder["domain"] == "domain"]

        # texts with similar token lengths are grouped together
        # to better approximate domain token length distribution
        # and truncate less from longer texts.
        grouped["group"] = grouped.groupby("label")["token_length"].rank(
            "first"
        )

        # truncate a text by setting its new length to:
        # min_token_length_per_group - min_tokens_to_truncate
        grouped["new_token_length"] = (
            grouped.groupby("group")["token_length"].transform(min)
            - min_tokens_to_truncate
        )
        grouped = grouped.drop(["group"], axis=1)

        # compute mean and std to tuncate the ungrouped texts
        # for this we must only consider new lengths within desired min and max
        new_lengths_above_min = grouped["new_token_length"][
            (grouped["new_token_length"] >= min_length)
        ]
        mean, std = new_lengths_above_min.mean(), new_lengths_above_min.std()

        # Sample lengths around mean +- 2 std
        low = int(max(min_length, mean - sampling_radius_size * std))
        high = int(mean + sampling_radius_size * std)
        sampled_new_token_lengths = np.random.randint(
            low=low, high=high, size=len(remainder)
        )

        # account for cases where the sampled token length > current token length
        remainder["new_token_length"] = np.minimum(
            sampled_new_token_lengths,
            remainder["token_length"] - min_tokens_to_truncate,
        )

        both = pd.concat([grouped, remainder])

        # drop rows not within length bounds
        both = both[(both["new_token_length"] >= min_length)]

        to_truncate.append(both)

    new_df = pd.concat(to_truncate).reset_index(drop=True)

    assert (new_df["token_length"] < new_df["new_token_length"]).sum() == 0

    def truncate_and_decode_one(row: Dict) -> str:
        tokenized_truncated = row["tokenized"][: row["new_token_length"]]
        text = "".join([token.text_with_ws for token in tokenized_truncated])
        return text

    new_df["text"] = new_df.apply(truncate_and_decode_one, axis=1)

    diff = new_df["token_length"] - new_df["new_token_length"]
    dropped_quantity = len(df) - len(new_df)

    _logger.info(
        f"Truncated texts. Length difference statistics: {diff.describe().to_dict()}"
    )
    if dropped_quantity:
        _logger.info(
            f"{dropped_quantity} texts were too short and were dropped in the truncation process."
        )

    new_df = new_df.drop(
        ["token_length", "new_token_length", "tokenized"], axis=1
    )

    dataset = Dataset.from_pandas(new_df)
    return dataset


def filter_by_language(dataset: Dataset, language: str = "en") -> Dataset:
    """
    Applies a language id filter, removing texts in undesired languages.

    Args:
        dataset (Dataset): the dataset to apply the language id filter on.
    Returns:
        Dataset: the filtered dataset.
    """
    model = get_langid_model()

    @batched_map
    def annotate_texts_with_language(texts: List[str]):
        keep = []
        for text in texts:
            # need to replace newlines since fasttexts doesn't accept them
            predicted_language = model.predict(text.replace("\n", " "), k=1)

            # fasttext model output looks like: (('__label__en',), array([0.98803425]))
            # so we grab the suffix with the language ISO code
            predicted_language = predicted_language[0][0][-2:]
            if predicted_language != language:
                keep.append(False)
            else:
                keep.append(True)

        return {"keep": keep}

    dataset = annotate_texts_with_language(dataset)
    dataset = dataset.filter(lambda x: x["keep"], desc="Filter by language")
    dataset = dataset.remove_columns(["keep"])

    return dataset


def get_langid_model() -> fasttext.FastText:
    url = (
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    )
    model_path = get_cache_path() / "fasttext" / "lid.176.bin"

    # Need to download if it doesn't exist
    try:
        model = fasttext.load_model(str(model_path))
    except (ValueError, FileNotFoundError):
        model_path.parent.mkdir(parents=True, exist_ok=True)
        content = requests.get(url, stream=True).content
        with model_path.open("wb") as f:
            f.write(content)

        model = fasttext.load_model(str(model_path))

    return model


def remove_label_duplicates(dataset: Dataset) -> Dataset:
    """
    Removes text with more than one associated label.

    Args
        dataset (Dataset): the dataset to remove label duplicates from.

    Returns:
        Dataset: the dataset with removed label duplicates.
    """
    old_len = len(dataset)
    df = dataset.to_pandas()

    groups = df.groupby("text")["label"].nunique()
    single_label_texts = groups[groups == 1].index
    filtered = df[df["text"].isin(single_label_texts)]

    dataset = Dataset.from_pandas(filtered, preserve_index=False)

    new_len = len(dataset)
    _logger.info(f"Removed {old_len - new_len} texts with more than one label.")

    return dataset


def remove_text_duplicates(dataset: Dataset) -> Dataset:
    """
    Removes all text duplicates.

    Args
        dataset (Dataset): the dataset to remove duplicates from.

    Returns:
        Dataset: the dataset with removed duplicates.
    """
    old_len = len(dataset)

    def hash_str(s: str) -> str:
        x = s.encode("utf8")
        return hashlib.sha256(x).hexdigest()

    dataset = dataset.map(
        lambda example: {"hash": hash_str(example["text"])},
        load_from_cache_file=False,
    )

    _, unique_indices = np.unique(dataset["hash"], return_index=True)

    dataset = dataset.select(unique_indices)

    dataset = dataset.remove_columns(["hash"])
    new_len = len(dataset)
    _logger.info(f"Removed {old_len - new_len} duplicated texts.")

    return dataset


def remove_empty_texts(dataset: Dataset) -> Dataset:
    """
    Removes empty texts from a dataset.

    Args:
        dataset (Dataset): the dataset to remove empty texts from.

    Returns:
        Datset: the dataset with removed empty texts.
    """
    old_len = len(dataset)
    dataset = dataset.filter(
        lambda example: example["text"], load_from_cache_file=False
    )
    new_len = len(dataset)
    _logger.info(f"Removed {old_len - new_len} empty texts.")
    return dataset


@batched_map
def remove_special_tokens(texts: List[str]) -> Dict[str, List[str]]:
    """
    Removes special text generation tokens from a list of texts.

    Args:
        texts (List[str]): the texts to apply special-token removal to.
    Returns:
        Dict[str, List[str]]: the cleaned texts in dict form.
        The result is returned as so in order to run this using
        batched mapping from huggingface datasets.
    """
    special_tokens = [
        "[CLS]",
        "[SEP]",
        "[PAD]",
        "[MASK]",
        "[UNK]",
        "[BOS]",
        "[EOS]",
        "[EOD]",
        "[EOP]",
        "<endoftext>",
    ]
    # Also remove brackets from ends
    special_tokens += [x[1:-1] for x in special_tokens]

    regex = re.compile("|".join(map(re.escape, special_tokens)))
    clean = []
    for text in texts:
        text = regex.sub("", text)
        clean.append(text)

    return {"text": clean}


@batched_map
def remove_disclosure_phrases(texts: List[str]) -> Dict[str, List[str]]:
    """
    Removes a set of disclosure phrases.

    Args:
        texts (List[str]): the texts from which to remove disclosure phrases.
    Returns:
        Dict[str, List[str]]: the cleaned texts in dict form.
        The result is returned as so in order to run this using
        batched mapping from huggingface datasets.
    """

    # These patterns match the prefixes until next word
    patterns = [
        r"As an AI language model.+?(?=\w)",
        r"I am sorry, I am an AI language model and.+?(?=\w)",
        r"I am sorry, I'm an AI language model and.+?(?=\w)",
        r"I'm sorry, I'm an AI language model and.+?(?=\w)",
        r"I'm sorry, I am an AI language model and.+?(?=\w)",
        r"I'm sorry, but I am an AI language model and.+?(?=\w)",
    ]
    regexes = [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]

    clean = []
    for text in texts:
        original_text = text

        for regex in regexes:
            text = regex.sub("", text)

        # At least one regex was applied so we correct the capitalization
        if original_text != text:
            text = text.capitalize()

        clean.append(text)

    return {"text": clean}


@batched_map
def fix_encoding(texts: List[str]) -> Dict[str, List[str]]:
    """
    Fixes the encoding in a list of texts.

    Args:
        texts (List[str]): the texts to apply encoding-fixing to.
    Returns:
        Dict[str, List[str]]: the cleaned texts in dict form.
        The result is returned as so in order to run this using
        batched mapping from huggingface datasets.
    """
    clean = []
    for text in texts:
        text = ftfy.fix_text(text)
        clean.append(text)

    return {"text": clean}


@batched_map
def strip(texts: List[str]) -> Dict[str, List[str]]:
    """
    Strips whitespace from a list of texts.

    Args:
        texts (List[str]): the texts to apply stripping to.
    Returns:
        Dict[str, List[str]]: the cleaned texts in dict form.
        The result is returned as so in order to run this using
        batched mapping from huggingface datasets.
    """
    clean = []
    for text in texts:
        text = text.strip()
        clean.append(text)
    return {"text": clean}


def postprocess(dataset: Dataset, task_type: TaskType) -> Dataset:
    """
    Postprocesses a dataset.

    Args:
        dataset (Dataset): the dataset to postprocess.
    Returns:
        Dataset: the postprocessed dataset.
    """

    single_text_actions = [
        fix_encoding,
        strip,
        remove_special_tokens,
        remove_disclosure_phrases,
    ]

    full_dataset_actions = [
        remove_generation_errors,
        remove_empty_texts,
        remove_text_duplicates,
    ]

    if task_type in {TaskType.DETECTION, TaskType.ATTRIBUTION}:
        if len(set(dataset["label"])) < 2:
            _logger.info(
                "Dataset only has single label, truncation will not be applied."
            )
            actions = (
                single_text_actions
                + full_dataset_actions
                + [remove_label_duplicates]
            )
        else:
            actions = (
                single_text_actions
                + [truncate]
                + full_dataset_actions
                + [remove_label_duplicates]
            )
    else:
        actions = single_text_actions + full_dataset_actions  # type: ignore

    for action in actions:
        dataset = action(dataset)  # type: ignore

    return dataset
