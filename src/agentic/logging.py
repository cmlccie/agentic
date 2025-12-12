"""Logging configurations and utilities."""

import logging
from functools import wraps
from typing import Callable

from rich.logging import RichHandler

# -------------------------------------------------------------------------------------------------
# Logging Configurations
# -------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# Silent
# --------------------------------------------------------------------------------------


def silent(level: int = logging.WARNING):
    """Remove all existing logging handlers and set the logging level."""

    # Remove all existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set the logging level
    root_logger.setLevel(level)

    # Add a null handler to prevent messages from going to console
    null_handler = logging.NullHandler()
    root_logger.addHandler(null_handler)


# --------------------------------------------------------------------------------------
# Fancy
# --------------------------------------------------------------------------------------


def fancy(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        handlers=[RichHandler()],
    )


# -------------------------------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# Function Decorator to Log Function Calls
# --------------------------------------------------------------------------------------


def log_call(logger: logging.Logger) -> Callable:
    """Decorator to log function calls and their results."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(
                f"Call: {func.__name__}({', '.join(map(repr, args))}, {', '.join(f'{k}={v!r}' for k, v in kwargs.items())})"
            )
            result = func(*args, **kwargs)
            logger.info(
                f"Result: {func.__name__}({', '.join(map(repr, args))}, {', '.join(f'{k}={v!r}' for k, v in kwargs.items())}) -> {result}"
            )
            return result

        return wrapper

    return decorator
