"""Tests for the MCP Customer Database Server (no real PostgreSQL)."""

import pytest
from fastmcp.exceptions import ToolError

from .helpers import call_tool, list_tool_names, load_server


@pytest.fixture(scope="module")
def db():
    return load_server("customer_database_server")


# -------------------------------------------------------------------------------------------------
# Query guard (pure functions)
# -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "  select * from customers",
        "WITH t AS (SELECT 1) SELECT * FROM t",
        "EXPLAIN SELECT * FROM customers",
        "-- top customers\nSELECT * FROM customers",
        "/* multi\nline */ /* two */ WITH x AS (SELECT 1) SELECT * FROM x",
        "SELECT(1)",
    ],
)
def test_read_only_statements_are_allowed(db, sql):
    assert db.check_read_only(sql) == sql


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM customers",
        "UPDATE customers SET name = 'x'",
        "-- SELECT\nDROP TABLE customers",
        "/* SELECT */ INSERT INTO customers VALUES (1)",
        "SELECTED",
        "",
        "   ",
        "-- only a comment",
    ],
)
def test_other_statements_are_rejected(db, sql):
    with pytest.raises(ValueError, match="read-only"):
        db.check_read_only(sql)


def test_leading_keyword(db):
    assert db.leading_keyword("/* c */ -- d\n  with x as (select 1) select 1") == "WITH"


# -------------------------------------------------------------------------------------------------
# query_database with a fake connection
# -------------------------------------------------------------------------------------------------


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.executed: list[str] = []
        self.description = [("id",), ("name",)]

    def execute(self, sql):
        self.executed.append(sql)

    def fetchmany(self, size):
        return self.rows[:size]


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.closed = False

    def cursor(self):
        return self._cursor

    def close(self):
        self.closed = True


@pytest.fixture
def fake_db(db, monkeypatch):
    def install(rows):
        cursor = FakeCursor(rows)
        connection = FakeConnection(cursor)
        monkeypatch.setattr(db, "_connect", lambda: connection)
        return cursor, connection

    return install


def test_tools_are_registered(db):
    assert list_tool_names(db.mcp) == {"get_schema", "query_database"}


def test_query_runs_read_only_with_timeout(db, fake_db, monkeypatch):
    monkeypatch.setattr(db, "QUERY_TIMEOUT_S", 2.5)
    cursor, connection = fake_db([(1, "Ada"), (2, "Grace")])

    result = call_tool(db.mcp, "query_database", {"sql": "WITH c AS (SELECT 1) SELECT"})

    assert result == {
        "rows": [{"id": 1, "name": "Ada"}, {"id": 2, "name": "Grace"}],
        "truncated": False,
    }
    assert cursor.executed == [
        "SET TRANSACTION READ ONLY",
        "SET LOCAL statement_timeout = 2500",
        "WITH c AS (SELECT 1) SELECT",
    ]
    assert connection.closed


def test_query_rows_are_capped(db, fake_db, monkeypatch):
    monkeypatch.setattr(db, "QUERY_MAX_ROWS", 2)
    fake_db([(i, f"n{i}") for i in range(10)])

    result = call_tool(db.mcp, "query_database", {"sql": "SELECT * FROM customers"})

    assert len(result["rows"]) == 2
    assert result["truncated"] is True


def test_rejected_query_never_connects(db, monkeypatch):
    def fail():
        raise AssertionError("should not connect")

    monkeypatch.setattr(db, "_connect", fail)
    with pytest.raises(ToolError, match="read-only"):
        call_tool(db.mcp, "query_database", {"sql": "DROP TABLE customers"})
