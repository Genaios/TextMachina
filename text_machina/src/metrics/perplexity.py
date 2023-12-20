from pathlib import Path

import evaluate
import pandas as pd
from datasets import Dataset

from ..common.exceptions import InvalidTaskTypeForMetric
from ..types import TaskType
from .base import Metric


class PerplexityMetric(Metric):
    """
    Implements the perplexity metric.
    """

    def _run(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        metric = evaluate.load("perplexity", module_type="metric")

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

        result = metric.compute(predictions=text, **kwargs)

        df = pd.DataFrame(
            {"label": label, "perplexity": result["perplexities"]}
        )

        return df

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
            logger.info(f"Mean perplexity of segments: {summary.to_dict()}")
        else:
            logger.info(f"Mean perplexity: {summary.to_dict()}")
