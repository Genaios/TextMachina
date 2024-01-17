from typing import Any, Dict, List

from datasets import Dataset
from rich.markup import escape
from rich.panel import Panel

from .base import Explorer


class MixCaseExplorer(Explorer):
    def __init__(self, dataset: Dataset) -> None:
        super().__init__(dataset)

    def get_panels(self, example: Dict[str, Any]) -> List[Panel]:
        panels = []
        # Prompt panels
        for idx, prompt in enumerate(example["prompt"]):
            panels.append(
                Panel(
                    escape(prompt),
                    title=f"[red3]prompt {idx}",
                    border_style="red3",
                    style="white",
                )
            )
        # Text panel
        text = ""
        for label in example["label"]:
            start, end = label["start"], label["end"]
            color = "[red3]" if label["label"] == "human" else "[blue3]"
            text += color + escape(example["text"][start : end + 1])
        panels.append(
            Panel(
                text,
                title="[turquoise2]text",
                border_style="turquoise2",
                style="white",
            )
        )
        return panels

    def get_title(self, idx: int, example: Dict[str, Any]) -> str:
        model = example["model"]
        domain = example["domain"]
        extractor = example["extractor"]

        title = f"[tan]{idx}[/]: "
        title += f"[violet]{model}[/] text "
        title += f"in domain [sea_green3]{domain}[/]"
        title += f" using extractor [yellow]{extractor}"

        return title
