"""Logging configurations and utilities."""

import functools
import inspect
import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

# -------------------------------------------------------------------------------------------------
# Logging Configurations
# -------------------------------------------------------------------------------------------------

# Both configurations replace the root logger's handlers (``force=True``), so calling
# either one repeatedly is idempotent and the last call wins.


# --------------------------------------------------------------------------------------
# Silent
# --------------------------------------------------------------------------------------


def silent(level: int | str = logging.WARNING) -> None:
    """Replace all root logging handlers with a null handler and set the level."""
    logging.basicConfig(level=level, handlers=[logging.NullHandler()], force=True)


# --------------------------------------------------------------------------------------
# Fancy
# --------------------------------------------------------------------------------------


def fancy(level: int | str = logging.INFO) -> None:
    """Replace all root logging handlers with a Rich handler and set the level.

    Logs are written to stderr so they never corrupt a stdio protocol stream (for
    example, an MCP server running with the ``stdio`` transport).
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        handlers=[RichHandler(console=Console(stderr=True))],
        force=True,
    )


# -------------------------------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# Function Decorator to Log Function Calls
# --------------------------------------------------------------------------------------

DEFAULT_REDACTED_NAMES: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "auth",
        "authorization",
        "client_secret",
        "credentials",
        "password",
        "passwd",
        "refresh_token",
        "secret",
        "token",
    }
)
"""Parameter names (case-insensitive, exact match) whose values are never logged."""

DEFAULT_MAX_REPR_LENGTH = 200
REDACTED = "'***'"


def _truncate(text: str, max_length: int) -> str:
    """Shorten ``text`` to ``max_length`` characters, marking any truncation."""
    if len(text) <= max_length:
        return text
    return f"{text[:max_length]}...<{len(text) - max_length} more chars>"


def _format_value(
    name: str, value: Any, redacted: frozenset[str], max_length: int
) -> str:
    """Return a safe, bounded repr of an argument value."""
    if name.lower() in redacted:
        return REDACTED
    return _truncate(repr(value), max_length)


def _named_arguments(
    signature: inspect.Signature | None, args: tuple, kwargs: Mapping[str, Any]
) -> Iterable[tuple[str, Any]]:
    """Pair every call argument with its parameter name (flattening ``**kwargs``)."""
    if signature is None:
        return [*((f"arg{i}", v) for i, v in enumerate(args)), *kwargs.items()]
    try:
        bound = signature.bind_partial(*args, **kwargs)
    except TypeError:
        return [*((f"arg{i}", v) for i, v in enumerate(args)), *kwargs.items()]
    return [
        pair
        for name, value in bound.arguments.items()
        for pair in (
            value.items()
            if signature.parameters[name].kind is inspect.Parameter.VAR_KEYWORD
            else [(name, value)]
        )
    ]


def log_call(
    logger: logging.Logger,
    level: int = logging.DEBUG,
    *,
    redact: Iterable[str] = DEFAULT_REDACTED_NAMES,
    max_repr_length: int = DEFAULT_MAX_REPR_LENGTH,
    log_result: bool = True,
) -> Callable[[Callable], Callable]:
    """Decorator to log function calls, their results, and raised exceptions.

    Works with both regular and ``async`` functions. Messages are only formatted
    when ``logger`` is enabled for ``level``, and they report the caller's location
    rather than this decorator's.

    Args:
        logger: The logger instance to use for logging.
        level: The logging level to use (default: logging.DEBUG).
        redact: Parameter names whose values are replaced with ``'***'``
            (case-insensitive exact match; default: common secret names such as
            ``api_key``, ``password``, ``token``, ``secret``, and ``authorization``).
        max_repr_length: Maximum length of each logged argument or result repr.
        log_result: Set to False to omit the return value from the log (for
            example, when results may contain personal data).

    Returns:
        A decorator that wraps the function while preserving its metadata and
        signature.
    """
    redacted = frozenset(name.lower() for name in redact)

    def decorator(func: Callable) -> Callable:
        name = func.__qualname__
        try:
            signature: inspect.Signature | None = inspect.signature(func)
        except (TypeError, ValueError):
            signature = None

        def describe(args: tuple, kwargs: Mapping[str, Any]) -> str:
            return ", ".join(
                f"{arg_name}={_format_value(arg_name, value, redacted, max_repr_length)}"
                for arg_name, value in _named_arguments(signature, args, kwargs)
            )

        # stacklevel=3: _log -> wrapper -> caller of the decorated function.
        def log_start(args: tuple, kwargs: Mapping[str, Any]) -> None:
            if logger.isEnabledFor(level):
                logger.log(
                    level, "Call: %s(%s)", name, describe(args, kwargs), stacklevel=3
                )

        def log_end(result: Any) -> None:
            if logger.isEnabledFor(level):
                shown = (
                    _truncate(repr(result), max_repr_length)
                    if log_result
                    else "<result not logged>"
                )
                logger.log(level, "Result: %s -> %s", name, shown, stacklevel=3)

        def log_error(error: BaseException) -> None:
            if logger.isEnabledFor(level):
                logger.log(
                    level,
                    "Raised: %s -> %s: %s",
                    name,
                    type(error).__name__,
                    _truncate(str(error), max_repr_length),
                    stacklevel=3,
                )

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                log_start(args, kwargs)
                try:
                    result = await func(*args, **kwargs)
                except Exception as error:
                    log_error(error)
                    raise
                log_end(result)
                return result

            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            log_start(args, kwargs)
            try:
                result = func(*args, **kwargs)
            except Exception as error:
                log_error(error)
                raise
            log_end(result)
            return result

        return wrapper

    return decorator
