from typing import Any, Dict

from datasets import Dataset
from readchar import key, readkey
from rich import print as rprint
from rich.console import Group
from rich.markup import escape
from rich.panel import Panel

from .types import TaskType


def step(dataset: Dataset, task_type: TaskType) -> None:
    """
    Steps through a generated dataset, handling user input and
    showing the texts in a command line UI.

    Args:
        dataset (Dataset): the dataset to step through
        task_type (TaskType): the type of task. UI formatting changes
            based on this.
    """
    idx = 0
    N = len(dataset)

    rich_print(idx, dataset[idx], task_type)

    while idx < len(dataset):
        idx = get_next_position_or_exit(idx, max_position=N - 1)
        rich_print(idx, dataset[idx], task_type)


def show_prompt() -> None:
    prompt_panel = Panel(
        "Prev: [yellow3]left arrow[/], [yellow3]A[/], or [yellow3]backspace[/].\n"
        "Next: [yellow3]right arrow[/], [yellow3]D[/], or [yellow3]enter[/].\n"
        "Quit: [yellow3]q[/].",
        expand=False,
    )
    rprint(prompt_panel)


def get_next_position_or_exit(position: int, max_position: int) -> int:
    """
    Handles user input to cycle through a dataset

    Args:
        current_position (int): the current position.
    Returns:
        int: the new position.
    """
    user_input = readkey()

    if user_input in {key.LEFT, "a", "A", key.BACKSPACE, key.DELETE}:
        position = max(position - 1, 0)
    elif user_input in {key.RIGHT, "d", "D", key.ENTER, key.CR, key.TAB}:
        position = min(position + 1, max_position)
    elif user_input in {"q", "Q"}:
        exit()

    return position


def rich_print(idx: int, row: Dict[str, Any], task_type: TaskType) -> None:
    """
    Prints a given row of a generated dataset as a pretty UI.

    Args:
        row (Dict[str, Any]): a row of a generated dataset.
        task_type (TaskType): the type of task. UI formatting changes
            based on this.
    """
    print("\033[H\033[J", end="")
    title = get_formatted_title(idx, row, task_type)

    panels = []
    if (
        task_type in {TaskType.DETECTION, TaskType.ATTRIBUTION}
        and row["label"] != "human"
    ):
        prompt_panel = Panel(
            escape(row["prompt"]),
            title="[red3]prompt",
            border_style="red3",
            style="white",
        )
        panels.append(prompt_panel)

    if task_type == TaskType.BOUNDARY:
        human = row["text"][: row["label"]]
        generated = row["text"][row["label"] :]
        text = f"[red3]{escape(human)}[blue3]{escape(generated)}"
    else:
        text = escape(row["text"])

    text_panel = Panel(
        text,
        title="[turquoise2]text",
        border_style="turquoise2",
        style="white",
    )
    panels.append(text_panel)

    rprint(Panel(Group(*panels), title=title))

    show_prompt()


def get_formatted_title(
    idx: int, row: Dict[str, Any], task_type: TaskType
) -> str:
    """
    Formats a panel title based on row data and the type of task.

    Args:
        row (Dict[str, Any]): a row of a generated dataset.
        task_type (TaskType): the type of task. UI formatting changes
            based on this.
    Returns:
        str: the formatted title.
    """
    prompt, text, label, model, domain, extractor, config = row.values()

    title = f"[tan]{idx}[/]: "

    if task_type == TaskType.DETECTION:
        if label == "generated":
            title += f"[violet]{label}[/] text ([light_coral]{model}[/]) "
        else:
            title += f"[violet]{label}[/] text "
    else:
        title += f"[violet]{model}[/] text "

    title += f"in domain [sea_green3]{domain}[/]"

    if label != "human":
        title += f" using extractor [yellow]{extractor}"

    return title
