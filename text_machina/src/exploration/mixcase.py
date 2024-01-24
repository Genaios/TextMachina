from typing import Any, Dict, List

from datasets import Dataset
from rich.markup import escape
from rich.panel import Panel

from .base import Explorer


class MixCaseExplorer(Explorer):
    """
    Explorer for mixcase tasks.
    """

    def __init__(self, dataset: Dataset) -> None:
        super().__init__(dataset)

    def get_panels(self, example: Dict[str, Any]) -> List[Panel]:
        panels = []
        # Prompt panels
        prompts = (
            [example["prompt"]]
            if not isinstance(example["prompt"], list)
            else example["prompt"]
        )
        for idx, prompt in enumerate(prompts):
            panels.append(
                Panel(
                    escape(prompt),
                    title=f"[red3]Prompt {idx}",
                    border_style="turquoise2",
                    style="white",
                )
            )
        # Text panel
        text = ""
        for label in example["label"]:
            start, end = label["start"], label["end"]
            color = "[red3]" if label["label"] == "human" else "[blue3]"
            text += color + escape(example["text"][start:end])

        model = example["model"]
        title = (
            "[red3]Human[/] text"
            if model == "human"
            else f"Mixcase text from [blue3]{model}[/] and [red3]human[/]"
        )
        panels.append(
            Panel(
                text,
                title=title,
                border_style="turquoise2",
                style="white",
            )
        )
        return panels

    def get_title(self, idx: int, example: Dict[str, Any]) -> str:
        example["model"]
        domain = example["domain"]
        extractor = example["extractor"]

        title = f"[tan]{idx}[/]: "
        title += f"Mixcase text in domain [sea_green3]{domain}[/]"
        title += f" using extractor [yellow]{extractor}"

        return title
