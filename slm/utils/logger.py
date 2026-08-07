"""
Structured logging module for SLM ecosystem.
"""

import logging
import sys
from typing import Optional


def get_logger(name: str = "slm", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Constructs and returns a configured logger instance.

    Args:
        name: Name of the logger module.
        log_file: Optional filepath to write log output.
        level: Logging severity level.

    Returns:
        logging.Logger: Standard python logger with custom formatting.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
