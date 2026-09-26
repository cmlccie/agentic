"""Tests for agentic.logging."""

import asyncio
import inspect
import logging

import pytest

import agentic.logging
from agentic.logging import REDACTED, log_call

logger = logging.getLogger("tests.log_call")


@pytest.fixture
def caplog_debug(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.DEBUG, logger=logger.name)
    return caplog


# -------------------------------------------------------------------------------------------------
# log_call
# -------------------------------------------------------------------------------------------------


def test_sync_call_and_result_are_logged(caplog_debug):
    @log_call(logger)
    def add(a: int, b: int = 2) -> int:
        return a + b

    assert add(1, b=3) == 4
    messages = [r.getMessage() for r in caplog_debug.records]
    assert messages == [
        "Call: test_sync_call_and_result_are_logged.<locals>.add(a=1, b=3)",
        "Result: test_sync_call_and_result_are_logged.<locals>.add -> 4",
    ]


def test_async_function_logs_awaited_result(caplog_debug):
    @log_call(logger)
    async def double(x: int) -> int:
        await asyncio.sleep(0)
        return x * 2

    assert inspect.iscoroutinefunction(double)
    assert asyncio.run(double(21)) == 42
    assert caplog_debug.records[-1].getMessage().endswith("double -> 42")
    assert "coroutine" not in caplog_debug.text


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [(("http://x", "s3cr3t"), {}), (("http://x",), {"api_key": "s3cr3t"})],
)
def test_sensitive_arguments_are_redacted(caplog_debug, args, kwargs):
    @log_call(logger)
    def probe(base_url: str, api_key: str | None = None, **extra: str) -> None:
        return None

    probe(*args, **kwargs, Password="hunter2")
    assert "s3cr3t" not in caplog_debug.text
    assert "hunter2" not in caplog_debug.text
    assert f"api_key={REDACTED}" in caplog_debug.text
    assert f"Password={REDACTED}" in caplog_debug.text
    assert "base_url='http://x'" in caplog_debug.text


def test_redaction_is_configurable_and_exact(caplog_debug):
    @log_call(logger, redact={"pin"})
    def f(pin: int, max_tokens: int) -> None:
        return None

    f(1234, max_tokens=512)
    assert "1234" not in caplog_debug.text
    assert "max_tokens=512" in caplog_debug.text


def test_long_reprs_are_truncated(caplog_debug):
    @log_call(logger, max_repr_length=10)
    def echo(value: str) -> str:
        return value

    echo("x" * 1000)
    assert "x" * 11 not in caplog_debug.text
    assert "more chars>" in caplog_debug.text


def test_result_can_be_omitted(caplog_debug):
    @log_call(logger, log_result=False)
    def secret_rows() -> list[str]:
        return ["alice@example.com"]

    secret_rows()
    assert "alice@example.com" not in caplog_debug.text
    assert "<result not logged>" in caplog_debug.text


def test_exceptions_are_logged_and_reraised(caplog_debug):
    @log_call(logger)
    def boom() -> None:
        raise RuntimeError("bad things")

    with pytest.raises(RuntimeError):
        boom()
    assert "Raised:" in caplog_debug.text
    assert "RuntimeError: bad things" in caplog_debug.text


def test_nothing_is_formatted_when_level_disabled(caplog):
    caplog.set_level(logging.INFO, logger=logger.name)
    reprs = []

    class Spy:
        def __repr__(self) -> str:
            reprs.append(1)
            return "Spy()"

    @log_call(logger, logging.DEBUG)
    def f(value: Spy) -> Spy:
        return value

    f(Spy())
    assert reprs == []
    assert caplog.records == []


def test_log_records_point_at_the_caller(caplog_debug):
    @log_call(logger)
    def f() -> None:
        return None

    @log_call(logger)
    async def g() -> None:
        return None

    async def awaiting_caller() -> None:
        await g()

    f()
    asyncio.run(awaiting_caller())
    assert [r.funcName for r in caplog_debug.records] == [
        "test_log_records_point_at_the_caller",
        "test_log_records_point_at_the_caller",
        "awaiting_caller",
        "awaiting_caller",
    ]


def test_metadata_and_signature_are_preserved():
    def original(a: int, api_key: str | None = None) -> int:
        """Docstring."""
        return a

    wrapped = log_call(logger)(original)
    assert wrapped.__name__ == "original"
    assert wrapped.__doc__ == "Docstring."
    assert inspect.signature(wrapped) == inspect.signature(original)


# -------------------------------------------------------------------------------------------------
# fancy / silent
# -------------------------------------------------------------------------------------------------


@pytest.fixture
def restore_root_logging():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield root
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.mark.parametrize("configure", [agentic.logging.fancy, agentic.logging.silent])
def test_configurations_are_idempotent(restore_root_logging, configure):
    root = restore_root_logging
    configure(logging.DEBUG)
    configure(logging.ERROR)
    assert len(root.handlers) == 1
    assert root.level == logging.ERROR
