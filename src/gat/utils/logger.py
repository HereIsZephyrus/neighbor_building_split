"""Logging utilities for GAT training."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Setup root logger with file and console handlers.

    This configures the root logger so that all child loggers (created via get_logger)
    will propagate their messages to the root logger's handlers.

    Args:
        name: Logger name (default: "" for root logger, recommended to ensure message propagation)
        log_file: Path to log file. If None, only console logging.
        level: Logging level

    Returns:
        Configured logger
    """
    # Use root logger if name is empty or "gat"
    if name == "gat":
        name = ""  # Force root logger for backward compatibility

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter with full module information
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if log_file is provided
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info("Logging to file: %s", log_file)

    return logger


def get_logger(name: str = "gat") -> logging.Logger:
    """
    Get logger instance that will propagate to root logger.

    This function should be called after setup_logger() has been called to configure
    the root logger. All messages from child loggers will propagate to the root logger's
    handlers (both console and file).

    Args:
        name: Logger name (usually __name__ from calling module for full module path)

    Returns:
        Logger instance (messages will propagate to root logger)
    """
    logger = logging.getLogger(name)

    # Don't add handlers - let messages propagate to root logger
    # The root logger should be configured via setup_logger() first

    # Set level if not already set (None means inherit from parent)
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)

    # Ensure propagation is enabled (it's True by default, but be explicit)
    logger.propagate = True

    return logger
