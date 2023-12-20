import logging
import sys
from datetime import datetime
from pathlib import Path

_time = datetime.now()


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
