from pathlib import Path
from typing import Dict, List

import evaluate
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from ..common.exceptions import InvalidTaskTypeForMetric
from ..types import TaskType
from .base import Metric


def prepare_tags(
    offset_mappings: List[List[List[int]]],
    labels: List[List[Dict]],
    label_mapping: Dict[str, int],
) -> Dict[str, List[List[int]]]:
    """
    Prepares the labels for token classification.
    This fn is designed to work with the `map` HF's function
    using `batched=True`.

    Args:
        offset_mappings (List[List[List[int]]]): offset mappings of each text.
        labels (List[List[Dict]]): labels of each text.
        label_mapping (Dict[str, int]): label mapping str to int labels.

    Returns:
        Dict[str, List[List[int]]]: the tags for each text in the batch.
    """
    tags = []
    for offset_mapping, label in zip(offset_mappings, labels):
        sample_tags = [-100]
        for _, char_end in offset_mapping[1:-1]:
            if char_end < label[0]["end"]:
                sample_tags.append(label_mapping[label[0]["label"]])
            else:
                if len(label) > 1:
                    label.pop(0)
                sample_tags.append(label_mapping[label[0]["label"]])
        sample_tags.append(-100)
        tags.append(sample_tags)
    return {"labels": tags}


def prepare_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    test_size: float,
    label_mapping: Dict[str, int],
) -> DatasetDict:
    """
    Prepares the dataset (tokenization, prepare labels, splitting, etc.).

    Args:
        dataset (Dataset): a dataset.
        tokenizer (AutoTokenizer): a tokenizer.
        test_size (float): proportion reserved for the test set.
        label_mapping (Dict[str, int]): label mapping str to int labels.

    Returns:
        DatasetDict: a dataset with train and test splits.
    """
    dataset = dataset.map(
        lambda batch: tokenizer(
            batch, truncation=True, return_offsets_mapping=True
        ),
        input_columns=["text"],
        batched=True,
    )
    dataset = dataset.map(
        prepare_tags,
        input_columns=["offset_mapping", "label"],
        batched=True,
        fn_kwargs={"label_mapping": label_mapping},
    )
    dataset = dataset.select_columns(["input_ids", "attention_mask", "labels"])
    dataset = dataset.train_test_split(test_size=test_size)
    return dataset


def fit(
    model: AutoModelForTokenClassification,
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    training_args: Dict,
) -> None:
    """
    Fits a model on a dataset.

    Args:
        model (AutoModelForTokenClassification): a model.
        dataset (Dataset): a training dataset.
        tokenizer (AutoTokenizer): a tokenizer.
        training_args (Dict): args to be passed to the HF's Trainer.
    """
    training_args = TrainingArguments(do_train=True, **training_args)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    trainer = Trainer(
        model,
        train_dataset=dataset,
        data_collator=collator,
        args=training_args,
    )
    trainer.train()


def predict(
    model: AutoModelForTokenClassification,
    dataset: Dataset,
    tokenizer: AutoTokenizer,
) -> List[List[int]]:
    """
    Predicts a dataset using a model.

    Args:
        model (AutoModelForTokenClassification): a model.
        dataset (Dataset): a test dataset.
        tokenizer (AutoTokenizer): a tokenizer.

    Returns:
        List[List[int]]: list of predicted labels for each example.
    """
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    trainer = Trainer(
        model,
        data_collator=collator,
    )
    predictions = trainer.predict(dataset).label_ids
    return predictions


def eval(
    predictions: List[List[int]],
    references: List[List[int]],
    label_mapping: Dict[int, str],
) -> Dict[str, float]:
    """
    Evaluates using `seqeval` from HF metrics.

    Args:
        predictions (List[List[int]]): list of predictions for each example.
        references (List[List[int]]): list of gold labels for each example.
        label_mapping (Dict[int, str]): label mapping int to str labels.

    Returns:
        Dict[str, float] -> dictionary with metric values.
    """
    preds = [
        [label_mapping[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, references)
    ]

    refs = [
        [label_mapping[l] for (_, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, references)
    ]

    preds = [pred[:-1] for pred in preds]

    seqeval = evaluate.load("seqeval")
    results = seqeval.compute(predictions=preds, references=refs)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


class TokenClassificationMetric(Metric):
    """
    Implements a HF token classification model
    for evaluating a mixcase dataset.
    """

    def _run(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        if self.task_type != TaskType.MIXCASE:
            raise InvalidTaskTypeForMetric(self.name, self.task_type)

        model = AutoModelForTokenClassification.from_pretrained(
            **kwargs["model_args"]
        )
        tokenizer = AutoTokenizer.from_pretrained(
            kwargs["model_args"]["pretrained_model_name_or_path"]
        )
        dataset = prepare_dataset(
            dataset, tokenizer, kwargs["test_size"], kwargs["label_mapping"]
        )

        fit(model, dataset["train"], tokenizer, kwargs["training_args"])
        predictions = predict(model, dataset["test"], tokenizer)

        results = eval(
            predictions,
            dataset["test"]["labels"],
            {v: k for k, v in kwargs["label_mapping"].items()},
        )
        return pd.DataFrame(results, index=[0])

    def _save(self, outputs: pd.DataFrame, path: Path) -> None:
        outputs.to_csv(path / "full_outputs.csv", index=False)

    def _log(self, outputs: pd.DataFrame, logger) -> None:
        logger.info(f"Seqeval results: {outputs}")
