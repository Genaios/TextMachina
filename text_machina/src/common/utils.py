import inspect
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional


def get_instantiation_args(
    fn: Callable, args: Dict, accepted_params: Optional[List[str]] = None
) -> Dict:
    """
    Extracts the arguments from `args` that match the
    parameters of the `fn` function.

    Args:
        fn (Callable): a function.
        args (Dict): a dictionary of arguments.
        accepted_params (Optional[List[str]]): if known, the list
            of params accepted by fn. If `accepted_params` is
            provided, the `fn` argument will not be used to infer
            the accepted parameters.
    Returns:
        Dict: arguments in `args` that match the `fn` parameters.
    """
    if accepted_params is None:
        accepted_params = list(inspect.signature(fn).parameters.keys())
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
