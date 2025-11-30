"""Logging utilities for the application."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Initialize root logger with console and optional file handlers.
    
    This function should be called ONCE at the start of the application.
    All child loggers created via get_logger() will inherit the root logger's
    level and handlers through propagation.
    
    Args:
        log_file: Optional path to log file. If None, only console logging.
        level: Logging level (default: INFO)
    
    Returns:
        Root logger instance
    
    Example:
        # At application start
        setup_logger(log_file=Path("app.log"), level=logging.DEBUG)
        
        # In any module
        logger = get_logger(__name__)  # Inherits DEBUG level automatically
    """
    # Always configure root logger (name="")
    root_logger = logging.getLogger()
    
    # Preserve more verbose level if root logger already configured
    # Lower numeric value = more verbose (DEBUG=10 < INFO=20)
    if root_logger.level != logging.NOTSET and root_logger.level < level:
        # Root logger already has more verbose level, keep it
        level = root_logger.level
    else:
        # Set root logger level
        root_logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)  # Hardcoded: only WARNING and above to console
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance for a module, inheriting root logger's configuration.
    
    This function should be used in all modules to get a logger. The logger
    will automatically inherit the root logger's level and handlers through
    propagation. No handlers are added to child loggers.
    
    Args:
        name: Logger name (usually __name__ from calling module)
    
    Returns:
        Logger instance that propagates to root logger
    
    Example:
        # In any module
        from .utils.logger import get_logger
        logger = get_logger(__name__)
        logger.debug("Debug message")  # Will use root logger's level
    """
    logger = logging.getLogger(name)
    
    # Get root logger to check configuration
    root_logger = logging.getLogger()
    
    # Set logger level based on root logger configuration
    if root_logger.level == logging.NOTSET:
        # Root logger not configured, use default INFO
        logger.setLevel(logging.INFO)
    else:
        # Root logger is configured, inherit its level
        logger.setLevel(root_logger.level)
    
    # Ensure propagation is enabled (messages bubble up to root logger handlers)
    logger.propagate = True
    
    # Do NOT add handlers to child loggers - use root logger's handlers via propagation
    # This ensures all loggers share the same handlers and level
    
    return logger
