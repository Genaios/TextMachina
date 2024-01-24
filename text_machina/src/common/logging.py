import logging
import sys
from datetime import datetime
from pathlib import Path

_time = datetime.now()

COLORS = {
    "grey": "\x1b[38;20m",
    "yellow": "\x1b[33;20m",
    "bold_yellow": "\x1b[33;1m",
    "red": "\x1b[31;20m",
    "bold_red": "\x1b[31;1m",
    "reset": "\x1b[0m",
}


def color_log(text: str, color: str) -> str:
    """
    Add color to a log text.

    Args:
        text (str): a text.
        color (str): a color in `COLORS`

    Returns:
        str: a text with color codes added.
    """
    return COLORS[color] + text + COLORS["reset"]


def get_logger(module_name: str) -> logging.Logger:
    """
    Returns the logger used across TextMachina modules.

    Args:
        module_name (str): name of the module.

    Returns:
        logging.Logger: the logger.
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logfile = (
        Path("logs")
        .joinpath(
            _time.strftime("%Y_%m_%d"),
            _time.strftime("%H_%M_%S"),
            "text-machina.log",
        )
        .absolute()
    )

    logfile.parent.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(logfile)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)

    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger
