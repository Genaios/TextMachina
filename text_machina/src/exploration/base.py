from abc import ABC, abstractmethod
from typing import Any, Dict, List

from datasets import Dataset
from readchar import key, readkey
from rich import print as rprint
from rich.console import Group
from rich.panel import Panel


class Explorer(ABC):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    @property
    def command_panel(self) -> Panel:
        """
        Command panel to help the users with commands.
        Returns:
            Panel: panel showing the commands and action keys.
        """
        return Panel(
            "Prev: [yellow3]left arrow[/], [yellow3]A[/], or [yellow3]backspace[/].\n"
            "Next: [yellow3]right arrow[/], [yellow3]D[/], or [yellow3]enter[/].\n"
            "Quit: [yellow3]q[/].",
            expand=False,
        )

    def step(self) -> None:
        """
        Steps through a generated dataset, handling user input and
        showing the texts in a command line UI.
        """
        idx = 0
        N = len(self.dataset)

        self.show_example(idx)

        while idx < len(self.dataset):
            idx = self.get_next_position_or_exit(idx, max_position=N - 1)
            self.show_example(idx)

    def get_next_position_or_exit(
        self, position: int, max_position: int
    ) -> int:
        """
        Handles user input to cycle through a dataset.

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

    def show_example(self, idx: int) -> None:
        """
        Pretty prints an example from the dataset.

        Args:
            idx: index of the example in the dataset.
        """
        print("\033[H\033[J", end="")
        # TODO: QUITAR EL TRY CUANDO LO PRUEBE TODO
        try:
            panels = self.get_panels(self.dataset[idx])
            title = self.get_title(idx, self.dataset[idx])
            rprint(Panel(Group(*panels), title=title))
            rprint(self.command_panel)
        except Exception as e:
            print(e)
            exit()

    @abstractmethod
    def get_panels(self, example: Dict[str, Any]) -> List[Panel]:
        """
        Builds the list of panels for the elements (prompts, texts, etc.)
        to be printed. This method should be implemented by each
        task-specific explorer.

        Args:
            example (Dict[str, Any]): an example from the dataset.
        Returns:
            List[Panel]: lists of panels to be printed in the desired order.
        """
        ...

    @abstractmethod
    def get_title(self, idx: int, example: Dict[str, Any]) -> str:
        """
        Builds a printable title based on an example from the dataset.
        This method should be implemented by each task-specific explorer.

        Args:
            idx: index of the example.
            example (Dict[str, Any]): an example from the dataset.
        Returns:
            str: a printable title.
        """
        ...
