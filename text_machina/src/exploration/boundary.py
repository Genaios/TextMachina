from typing import Any, Dict, List

from datasets import Dataset
from rich.markup import escape
from rich.panel import Panel

from .base import Explorer


class BoundaryExplorer(Explorer):
    """
    Explorer for boundary tasks.
    """

    def __init__(self, dataset: Dataset) -> None:
        super().__init__(dataset)

    def get_panels(self, example: Dict[str, Any]) -> List[Panel]:
        panels = []
        # Text panel
        human = example["text"][: example["label"]]
        generated = example["text"][example["label"] :]
        text = f"[red3]{escape(human)}[blue3]{escape(generated)}"
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
