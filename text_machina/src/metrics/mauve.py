import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import mauve
from datasets import Dataset

from ..common.exceptions import InvalidTaskTypeForMetric, UnsupportedMetricParam
from ..types import TaskType
from .base import Metric


class MAUVEMetric(Metric):
    """
    Implements the MAUVE metric: https://arxiv.org/abs/2102.01454

    Currently only supports the MGT detection task.
    """

    def _run(self, dataset: Dataset, **kwargs) -> Dict:
        if self.task_type != TaskType.DETECTION:
            raise InvalidTaskTypeForMetric(self.name, self.task_type)

        self.check_kwargs_ok(kwargs)

        df = dataset.to_pandas()

        generated_texts = df[df["label"] == "generated"]["text"]
        human_texts = df[df["label"] == "human"]["text"]

        outputs = mauve.compute_mauve(
            p_text=generated_texts, q_text=human_texts, **kwargs
        )

        return vars(outputs)

    def _save(self, outputs: Dict, path: Path) -> None:
        plt.plot(
            outputs["divergence_curve"][:, 1], outputs["divergence_curve"][:, 0]
        )
        plt.savefig(path / "divergence_curve.pdf")

        result = {}
        for k, v in outputs.items():
            if k in {"mauve", "frontier_integral"}:
                result[k] = v
            elif k in {"q_hist", "p_hist"}:
                result[k] = v.tolist()

        with open(path / "summary.json", "w") as f:
            json.dump(result, f, indent=4)

    def _log(self, outputs: Dict, logger) -> None:
        logger.info(f"MAUVE score: {outputs['mauve']}")
        logger.info(f"Frontier Integral: {outputs['frontier_integral']}")

    def check_kwargs_ok(self, kwargs) -> None:
        # Can't accept these since they are mutually exclusive with passing text as input
        unsupported = [
            x
            for x in {"p_tokens", "q_tokens", "p_features", "q_features"}
            if x in kwargs
        ]
        if unsupported:
            raise UnsupportedMetricParam(unsupported[0], self.name)
