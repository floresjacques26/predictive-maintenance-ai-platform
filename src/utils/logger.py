"""Centralized logging configuration."""

import logging
import sys
from pathlib import Path


def get_logger(name: str, level: int = logging.INFO, log_file: str | None = None) -> logging.Logger:
    """Create a structured logger with console and optional file handlers.

    Args:
        name: Logger name (typically __name__ of calling module).
        level: Logging level.
        log_file: Optional path to write logs to a file.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
