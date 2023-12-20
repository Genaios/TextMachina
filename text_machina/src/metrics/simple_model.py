from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import sklearn.metrics
import textstat
from datasets import Dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from tqdm import tqdm

from ..common.exceptions import InvalidTaskTypeForMetric
from ..types import TaskType
from .base import Metric


class SimpleModelMetric(Metric):
    """
    Implements simple baseline models evaluated with stratified k-fold validation.
    """

    def _run(self, dataset: Dataset, **kwargs) -> Dict[int, pd.DataFrame]:
        if self.task_type not in {
            TaskType.DETECTION,
            TaskType.ATTRIBUTION,
            TaskType.BOUNDARY,
        }:
            raise InvalidTaskTypeForMetric(self.name, self.task_type)

        df = dataset.to_pandas()

        if self.task_type == TaskType.BOUNDARY:
            kf = KFold(**kwargs.get("folds", {}))
        else:
            kf = StratifiedKFold(**kwargs.get("folds", {}))

        X, y = df["text"], df["label"]

        reports = {}
        iterator = tqdm(
            enumerate(kf.split(df, df["label"])),
            desc="Running simple model",
            total=kf.get_n_splits(),
        )

        for i, (train_idx, test_idx) in iterator:
            train_X, train_y = X.iloc[train_idx], y.iloc[train_idx]
            test_X, test_y = X.iloc[test_idx], y.iloc[test_idx]

            # Predict based on max readability differences for every prefix-suffix pair
            if self.task_type == TaskType.BOUNDARY:
                preds = self._predict_for_boundaries(test_X)
                report = regression_report(test_y, preds)
            # Train a BOW+BOC+LR model
            else:
                model = self._get_model(kwargs)
                model.fit(train_X, train_y)
                preds = model.predict(test_X)
                report = pd.DataFrame(
                    sklearn.metrics.classification_report(
                        test_y, preds, output_dict=True, zero_division=0.0
                    )
                )
            reports[i] = report

        return reports

    def _save(self, outputs: Dict[int, pd.DataFrame], path: Path) -> None:
        single_path = path / "single"
        single_path.mkdir(parents=True, exist_ok=True)

        for k, df in outputs.items():
            df.to_json(single_path / f"{k}_single.json", indent=4)
            df.to_markdown(single_path / f"{k}_single.md")

        aggregated = self._get_aggregated_results(outputs, std=True)

        aggregated.to_json(path / "aggregated.json", indent=4)
        aggregated.to_markdown(path / "aggregated.md")

    def _log(self, outputs: Dict[int, pd.DataFrame], logger) -> None:
        aggregated = self._get_aggregated_results(outputs, std=False)
        logger.info(f"Mean results:\n {aggregated}")

    def _get_aggregated_results(
        self, outputs: Dict[int, pd.DataFrame], std: bool = False
    ) -> pd.DataFrame:
        concatenated = pd.concat(outputs.values())

        if std:
            aggregated = (
                concatenated.reset_index().groupby("index").agg(["mean", "std"])
            )
            aggregated.columns = [
                "_".join(col) for col in aggregated.columns.values
            ]
        else:
            aggregated = concatenated.reset_index().groupby("index").mean()

        # Support must be added, not averaged
        totals = (
            pd.concat(outputs.values()).reset_index().groupby("index").sum()
        )
        if self.task_type == TaskType.BOUNDARY:
            if std:
                aggregated.loc[0, "support_mean"] = totals["support"].iloc[0]
                aggregated.loc[0, "support_std"] = 0.0
            else:
                aggregated.loc[0, "support"] = totals["support"].iloc[0]
        else:
            labels = [
                x
                for x in outputs[0].columns
                if x not in {"macro avg", "weighted avg", "accuracy"}
            ]
            for label in labels:
                if std:
                    aggregated.loc["support", f"{label}_mean"] = totals.loc[
                        "support"
                    ][label]
                    aggregated.loc["support", f"{label}_std"] = 0.0
                else:
                    aggregated.loc["support", label] = totals.loc["support"][
                        label
                    ]

        return aggregated

    def _get_model(self, kwargs) -> Pipeline:
        char_params = kwargs.get("feature_params", {}).get("char", {})
        word_params = kwargs.get("feature_params", {}).get("word", {})
        model_params = kwargs.get("model_params", {})

        return make_pipeline(
            FeatureUnion(
                [
                    ("char", CountVectorizer(analyzer="char", **char_params)),
                    (
                        "word",
                        CountVectorizer(analyzer="word", **word_params),
                    ),
                ]
            ),
            LogisticRegression(**model_params),
        )

    def _predict_for_boundaries(self, texts: List[str]) -> np.ndarray:
        """Predict position of maximal difference in readabilities"""
        preds = []
        for text in texts:
            result = []
            N = len(text)
            # prefixes and suffixes every 10 characters
            # running this without jumps would be too slow
            for i in range(20, N - 20, 10):
                prefix, suffix = text[:i], text[i:]
                prefix_score = textstat.flesch_reading_ease(prefix)
                suffix_score = textstat.flesch_reading_ease(suffix)

                # distance linearly weighed by index
                # weighing is necessary since otherwise the maximal difference
                # will always be between first prefix and suffix due to their
                # differences in length
                if i > N // 2:
                    weight = 2 - (i / (N // 2))
                else:
                    weight = i / (N // 2)

                result.append(abs(prefix_score - suffix_score) * weight)

            preds.append(result.index(max(result)))

        return np.array(preds)


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Computes a regression report similarly to the classification report
    offered by scikit-learn.

    Args:
        y_true (np.ndarray): the true labels.
        y_pred (np.ndarray): the predicted labels.
    Returns:
        pd.DataFrame: the report.
    """
    metrics = {
        "mean absolute error": [
            sklearn.metrics.mean_absolute_error(y_true, y_pred)
        ],
        "median absolute error": [
            sklearn.metrics.median_absolute_error(y_true, y_pred)
        ],
        "mean squared error": [
            sklearn.metrics.mean_squared_error(y_true, y_pred)
        ],
        "max error": [sklearn.metrics.max_error(y_true, y_pred)],
        "r2 score": [sklearn.metrics.r2_score(y_true, y_pred)],
        "explained variance score": [
            sklearn.metrics.explained_variance_score(y_true, y_pred)
        ],
        "support": [len(y_true)],
    }

    df = pd.DataFrame(metrics)
    return df
