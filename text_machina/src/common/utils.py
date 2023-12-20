import inspect
import os
from pathlib import Path
from typing import Callable, Dict


def get_instantiation_args(fn: Callable, args: Dict) -> Dict:
    """
    Extracts the arguments from `args` that match the
    parameters of the `fn` function.

    Args:
        fn (Callable): a function.
        args (Dict): a dictionary of arguments.
    Returns:
        Dict: arguments in `args` that match the `fn` parameters.
    """
    accepted_params = inspect.signature(fn).parameters
    return {arg: value for arg, value in args.items() if arg in accepted_params}


def get_cache_path() -> Path:
    """
    Get the TextMachina folder for caching intermediate results.

    Returns:
        Path: cache path.
    """
    path = Path(
        os.getenv("TEXT_MACHINA_CACHE_DIR", default="/tmp/text_machina_cache")
    )
    path.mkdir(parents=True, exist_ok=True)

    return path
