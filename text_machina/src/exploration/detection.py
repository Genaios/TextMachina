from typing import Any, Dict, List

from datasets import Dataset
from rich.markup import escape
from rich.panel import Panel

from .base import Explorer


class DetectionExplorer(Explorer):
    def __init__(self, dataset: Dataset) -> None:
        super().__init__(dataset)

    def get_panels(self, example: Dict[str, Any]) -> List[Panel]:
        panels = []
        # Prompt panel
        if example["label"] != "human":
            panels.append(
                Panel(
                    escape(example["prompt"]),
                    title="[red3]prompt",
                    border_style="red3",
                    style="white",
                )
            )

        # Text panel
        panels.append(
            Panel(
                escape(example["text"]),
                title="[turquoise2]text",
                border_style="turquoise2",
                style="white",
            )
        )
        return panels

    def get_title(self, idx: int, example: Dict[str, Any]) -> str:
        label, model = example["label"], example["model"]
        domain, extractor = example["domain"], example["extractor"]
        title = f"[tan]{idx}[/]: "
        if label == "generated":
            title += f"[violet]{label}[/] text ([light_coral]{model}[/]) "
        else:
            title += f"[violet]{label}[/] text "

        title += f"in domain [sea_green3]{domain}[/]"

        if label != "human":
            title += f" using extractor [yellow]{extractor}"

        return title
