import logging
from typing import Optional

import click

# --------------------------------------------------------------------------------------
# Silent Logging Config
# --------------------------------------------------------------------------------------


def silent_config(level: int = logging.WARNING) -> None:
    """Configure logging to disable console output."""

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
# Fancy Logging Config
# --------------------------------------------------------------------------------------


class ColoredFormatter(logging.Formatter):
    """Custom formatter that uses click for colored output."""

    # Color mapping for different log levels
    COLORS = {
        logging.DEBUG: "cyan",
        logging.INFO: "blue",
        logging.WARNING: "yellow",
        logging.ERROR: "red",
        logging.CRITICAL: "magenta",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors based on log level."""
        # Get the original formatted message
        message = super().format(record)

        # Get the color for this log level
        color = self.COLORS.get(record.levelno, "white")

        # Return the colored message
        return click.style(message, fg=color)


class ClickHandler(logging.Handler):
    """Custom logging handler that uses click.echo for output."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record using click.echo."""
        try:
            message = self.format(record)
            click.echo(message, err=True)
        except Exception:
            self.handleError(record)


def fancy_config(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    date_format: Optional[str] = None,
) -> None:
    """Configure logging with formatting and colored output by logging level.

    Args:
        level: The logging level threshold (default: logging.INFO)
        format_string: Custom format string (default: includes timestamp, level, name, and message)
        date_format: Custom date format string (default: ISO 8601 format)
    """

    # Default format if none provided
    if format_string is None:
        format_string = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"

    # Default date format if none provided
    if date_format is None:
        date_format = "%Y-%m-%dT%H:%M:%S"

    # Remove all existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set the logging level
    root_logger.setLevel(level)

    # Create the colored formatter
    formatter = ColoredFormatter(format_string, date_format)

    # Create the click handler
    handler = ClickHandler()
    handler.setFormatter(formatter)

    # Add the handler to the root logger
    root_logger.addHandler(handler)


# --------------------------------------------------------------------------------------
# Colorized Logging Config
# --------------------------------------------------------------------------------------


def colorized_config(level: int = logging.INFO) -> None:
    """Configure logging with colored output by logging level."""
    fancy_config(level, format_string="%(message)s")
