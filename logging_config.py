# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Centralized logging configuration for Heat Treatment Scheduler.

This module provides a unified logging setup for all components of the heat treatment
scheduler environment, including the client, server, and inference modules. It configures
logging to console and optionally to file, with consistent formatting across all modules.

Usage:
    >>> from logging_config import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Starting environment...")
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    """
    Get or create a logger with the specified name.

    This function provides a centralized way to create loggers across the project.
    All loggers use the same formatting and handlers configured in this module.

    Args:
        name: Logger name, typically __name__ of the calling module
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               Defaults to INFO or environment variable LOG_LEVEL

    Returns:
        logging.Logger: Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.debug("Debug message")
        >>> logger.info("Info message")
        >>> logger.warning("Warning message")
        >>> logger.error("Error message")
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured (avoid duplicate handlers)
    if not logger.handlers:
        # Determine log level from parameter, environment, or default
        if level is None:
            level = os.getenv("LOG_LEVEL", "INFO")

        logger.setLevel(level.upper())

        # Create formatters
        detailed_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level.upper())
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler (optional, if LOG_FILE environment variable is set)
        log_file = os.getenv("LOG_FILE")
        if log_file:
            # Create logs directory if it doesn't exist
            log_dir = Path(log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(level.upper())
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)

    return logger


def configure_logging(
    level: str = "INFO",
    log_file: str | None = None,
    format_type: str = "detailed"
) -> None:
    """
    Configure root logger and all Heat Treatment Scheduler loggers.

    This function sets up the logging system for the entire application,
    allowing you to control the log level, output file, and format globally.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: INFO
        log_file: Optional path to log file for persistent logging
        format_type: "detailed" (with function name and line number) or "simple"

    Example:
        >>> configure_logging(level="DEBUG", log_file="logs/hts.log")
        >>> logger = get_logger(__name__)
        >>> logger.debug("Detailed debug information")
    """
    # Set environment variables for get_logger to use
    os.environ["LOG_LEVEL"] = level.upper()
    if log_file:
        os.environ["LOG_FILE"] = log_file

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatters
    if format_type == "detailed":
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


# Module-level logger for this logging configuration module
_logger = logging.getLogger(__name__)
