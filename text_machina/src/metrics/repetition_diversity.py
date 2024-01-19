from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import Dataset

from ..common.exceptions import InvalidTaskTypeForMetric
from ..types import TaskType
from .base import Metric


class RepetitionDiversityMetric(Metric):
    """
    Implements the repetition and diversity metrics.
    See Sec. 4.1.2: https://arxiv.org/pdf/2202.06417.pdf

    Supported tasks: detection, attribution, and boundary.
    """

    def _run(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        # If we evaluate complete texts we can run easily grouping by labels
        if self.task_type in {TaskType.DETECTION, TaskType.ATTRIBUTION}:
            label = dataset["label"]
            text = dataset["text"]
        # Otherwise we need to separate the sections that pertain to different labels
        elif self.task_type == TaskType.BOUNDARY:
            df = dataset.to_pandas()[["text", "label"]]

            human_text = df.apply(
                lambda x: x["text"][: x["label"]], axis=1
            ).tolist()
            generated_text = df.apply(
                lambda x: x["text"][x["label"] :], axis=1
            ).tolist()

            text = human_text + generated_text
            label = ["human"] * len(human_text) + ["generated"] * len(
                generated_text
            )
        else:
            raise InvalidTaskTypeForMetric(self.name, self.task_type)

        ngrams = kwargs.get("ngrams", [2, 3, 4])

        results = []
        for text, label in zip(text, label):
            result = self.repetition_and_diversity(text, ngrams)
            result["label"] = label
            results.append(result)

        return pd.DataFrame(results)

    def _save(self, outputs: pd.DataFrame, path: Path) -> None:
        outputs.to_csv(path / "full_outputs.csv")
        outputs.drop("label", axis=1).mean(axis=0).to_json(
            path / "summary.json", indent=4
        )
        outputs.groupby("label").mean().reset_index().to_json(
            path / "per_label_summary.json", indent=4
        )

    def _log(self, outputs: pd.DataFrame, logger) -> None:
        summary = outputs.groupby("label").mean().reset_index()
        if self.task_type == TaskType.BOUNDARY:
            logger.info(
                f"Mean rep-n and diversity of segments: {summary.to_dict()}"
            )
        else:
            logger.info(f"Mean rep-n and diversity: {summary.to_dict()}")

    def repetition_and_diversity(self, text: str, ns: List[int]) -> Dict:
        tokens = text.strip().split()

        result = {}

        diversity = 1.0
        for n in ns:
            start_range = range(len(tokens) - n + 1)
            end_range = range(n, len(tokens) + 1)

            current_ngrams = [
                tuple(tokens[i:j]) for i, j in zip(start_range, end_range)
            ]

            uniques = len(set(current_ngrams))
            total = len(current_ngrams)

            if total > 0:
                ratio = uniques / total
            else:
                ratio = 0

            diversity *= ratio
            result[f"rep-{n}"] = 100 * (1.0 - (ratio))

        result["div"] = diversity

        return result
