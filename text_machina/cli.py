import atexit
import os
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from .cli_utils import _explore, _generate, generate_run_name, log_final_message
from .src.types import TaskType

os.environ["TOKENIZERS_PARALLELISM"] = "false"
app = typer.Typer()


@app.command()
def explore(
    config_path: Annotated[
        Path,
        typer.Option(
            help="The path to the YAML config or directory with YAML configs.",
            exists=True,
            file_okay=True,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    metrics_path: Annotated[
        Optional[Path],
        typer.Option(
            help="the path to the YAML metrics file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    run_name: Annotated[
        Optional[str],
        typer.Option(
            help="The name of the run. Generated automatically if none given."
        ),
    ] = None,
    save_dir: Annotated[
        Optional[Path],
        typer.Option(help="The path where the exploration will be saved."),
    ] = None,
    task_type: Annotated[
        TaskType, typer.Option(help="The type of task.")
    ] = TaskType.DETECTION,
    max_generations: Annotated[
        int,
        typer.Option(
            help="The maximum number of texts to generate per config."
        ),
    ] = 10,
    step: Annotated[
        bool, typer.Option(help="Whether to step through the generated texts")
    ] = True,
) -> None:
    """
    Generates a small set of texts, compares it against human texts,
    providing metrics and an interface for manual inspection of the generations
    """
    if not run_name:
        run_name = generate_run_name()
    if not save_dir:
        save_dir = Path.cwd() / run_name
    else:
        save_dir = save_dir / run_name

    atexit.register(log_final_message, run_name)

    _explore(
        config_path,
        metrics_path,
        save_dir,
        run_name,
        task_type,
        step,
        max_generations,
    )


@app.command()
def generate(
    config_path: Annotated[
        Path,
        typer.Option(
            help="The path to the YAML config or directory with YAML configs.",
            exists=True,
            file_okay=True,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    run_name: Annotated[
        Optional[str],
        typer.Option(
            help="The name of the run. Generated automatically if none given.",
        ),
    ] = None,
    save_dir: Annotated[
        Optional[Path],
        typer.Option(
            help=("The path where the generations will be saved."),
        ),
    ] = None,
    task_type: Annotated[
        TaskType, typer.Option(help="The type of task.")
    ] = TaskType.DETECTION,
) -> None:
    """
    Generates a dataset from the provided config or directory with configs.
    """
    if not run_name:
        run_name = generate_run_name()
    if not save_dir:
        save_dir = Path.cwd() / run_name
    else:
        save_dir = save_dir / run_name

    atexit.register(log_final_message, run_name)
    _generate(config_path, save_dir, run_name, task_type)


if __name__ == "__main__":
    app()
