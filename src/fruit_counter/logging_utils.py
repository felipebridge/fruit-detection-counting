"""Logging setup for command-line usage."""

from __future__ import annotations

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%H:%M:%S"


def configure_logging(level: str = "INFO") -> None:
    """Send package logs to stderr so stdout stays clean for results."""
    numeric = logging.getLevelName(level.upper())
    if not isinstance(numeric, int):
        raise ValueError(f"Unknown log level: {level}")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    package_logger = logging.getLogger("fruit_counter")
    package_logger.handlers[:] = [handler]
    package_logger.setLevel(numeric)
    package_logger.propagate = False
