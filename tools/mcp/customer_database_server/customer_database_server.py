#!/usr/bin/env python3
"""MCP Customer Database Server."""

import logging
import os
import re
from typing import Annotated, Any, Literal

import pg8000.dbapi
import typer
from fastmcp import FastMCP
from pydantic import BaseModel, Field

import agentic.logging

logger = logging.getLogger("customer_database_server")


HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

PGHOST = os.environ.get("PGHOST", "localhost")
PGPORT = int(os.environ.get("PGPORT", "5432"))
PGDATABASE = os.environ.get("PGDATABASE", "customers")
PGUSER = os.environ.get("PGUSER", "customers")
PGPASSWORD = os.environ.get("PGPASSWORD", "")

QUERY_TIMEOUT_S = float(os.environ.get("QUERY_TIMEOUT_S", "5"))
QUERY_MAX_ROWS = int(os.environ.get("QUERY_MAX_ROWS", "500"))

READ_ONLY_KEYWORDS = frozenset({"SELECT", "WITH", "EXPLAIN"})

# Leading whitespace, `-- line` comments, and `/* block */` comments.
_LEADING_NOISE = re.compile(r"\A(?:\s+|--[^\n]*(?:\n|\Z)|/\*.*?\*/)*", re.DOTALL)


# -------------------------------------------------------------------------------------------------
# MCP Customer Database Server
# -------------------------------------------------------------------------------------------------


mcp = FastMCP("MCP Customer Database Server")


def _connect() -> pg8000.dbapi.Connection:
    """Open a new connection to the customer database."""
    return pg8000.dbapi.connect(
        host=PGHOST,
        port=PGPORT,
        database=PGDATABASE,
        user=PGUSER,
        password=PGPASSWORD,
    )


# --------------------------------------------------------------------------------------
# Query Guard
# --------------------------------------------------------------------------------------


def leading_keyword(sql: str) -> str:
    """Return the first SQL keyword (upper-cased), ignoring leading comments."""
    body = _LEADING_NOISE.sub("", sql, count=1)
    match = re.match(r"[A-Za-z]+", body)
    return match.group(0).upper() if match else ""


def check_read_only(sql: str) -> str:
    """Validate that ``sql`` looks like a read-only statement and return it.

    This is a friendly first-line check for the LLM, not a security boundary: the
    query also runs in a ``READ ONLY`` transaction (which rejects data-modifying
    CTEs and ``EXPLAIN ANALYZE`` of writes), and the database role should only
    have ``SELECT`` privileges.

    Raises:
        ValueError: If the statement does not start with SELECT, WITH, or EXPLAIN.
    """
    keyword = leading_keyword(sql)
    if keyword not in READ_ONLY_KEYWORDS:
        raise ValueError(
            "Only read-only queries are allowed: the statement must start with "
            f"SELECT, WITH, or EXPLAIN (got {keyword or 'nothing'!r})."
        )
    return sql


# --------------------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------------------


@mcp.tool()
@agentic.logging.log_call(logger)
def get_schema() -> str:
    """Return a DDL-like description of all tables in the customer database.

    Use this first to discover the available tables and columns before writing a query.

    Returns:
        A human-readable schema listing each public table with its columns, data types,
        and nullability.
    """
    query = """
        SELECT table_name, column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position
    """
    conn = _connect()
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
    finally:
        conn.close()

    tables: dict[str, list[str]] = {}
    for table_name, column_name, data_type, is_nullable in rows:
        null = "NULL" if is_nullable == "YES" else "NOT NULL"
        tables.setdefault(table_name, []).append(
            f"    {column_name} {data_type} {null}"
        )

    return "\n\n".join(
        f"-- {table}\nCREATE TABLE {table} (\n" + ",\n".join(columns) + "\n);"
        for table, columns in tables.items()
    )


class QueryResult(BaseModel):
    """Rows returned by a read-only query."""

    rows: list[dict[str, Any]] = Field(
        ..., description="Result rows as dictionaries of column name to value."
    )
    truncated: bool = Field(
        ...,
        description="True when more rows matched than were returned; narrow the "
        "query or aggregate to see the rest.",
    )


@mcp.tool()
@agentic.logging.log_call(logger, log_result=False)
def query_database(sql: str) -> QueryResult:
    """Run a read-only SQL query against the customer database and return the rows.

    The statement must start with SELECT, WITH, or EXPLAIN; anything else is
    rejected. Queries run in a read-only transaction with a statement timeout, and
    at most a fixed number of rows are returned (`truncated` is true when rows were
    cut off). Prefer filters, aggregates, and LIMIT over fetching whole tables.

    Args:
        sql: A single read-only SQL statement to execute.

    Returns:
        QueryResult: The result rows and whether they were truncated.
    """
    check_read_only(sql)

    conn = _connect()
    try:
        cursor = conn.cursor()
        cursor.execute("SET TRANSACTION READ ONLY")
        cursor.execute(
            f"SET LOCAL statement_timeout = {max(1, int(QUERY_TIMEOUT_S * 1000))}"
        )
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description or []]
        rows = cursor.fetchmany(QUERY_MAX_ROWS + 1)
    finally:
        conn.close()

    logger.debug("query_database returned %d row(s)", min(len(rows), QUERY_MAX_ROWS))
    return QueryResult(
        rows=[dict(zip(columns, row, strict=True)) for row in rows[:QUERY_MAX_ROWS]],
        truncated=len(rows) > QUERY_MAX_ROWS,
    )


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------


def main(
    transport: Annotated[Literal["stdio", "http"], typer.Argument()] = "stdio",
) -> None:
    """Model Context Protocol (MCP) Customer Database Server."""
    agentic.logging.fancy()
    logger.info("Starting %s MCP Customer Database Server", transport)

    match transport:
        case "stdio":
            mcp.run(transport=transport)
        case "http":
            mcp.run(transport=transport, host=HOST, port=PORT)
        case _:
            raise typer.BadParameter(
                "Transport must be one of: stdio, http.",
                param_hint="transport",
            )


if __name__ == "__main__":
    typer.run(main)
