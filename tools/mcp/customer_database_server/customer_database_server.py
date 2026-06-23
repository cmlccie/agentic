#!/usr/bin/env python3
"""MCP Customer Database Server."""

import logging
import os
from typing import Annotated, Any, Literal

import pg8000.dbapi
import typer
from fastmcp import FastMCP

import agentic.logging

agentic.logging.fancy()
logger = logging.getLogger("customer_database_server")


HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

PGHOST = os.environ.get("PGHOST", "localhost")
PGPORT = int(os.environ.get("PGPORT", "5432"))
PGDATABASE = os.environ.get("PGDATABASE", "customers")
PGUSER = os.environ.get("PGUSER", "customers")
PGPASSWORD = os.environ.get("PGPASSWORD", "")


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


@mcp.tool()
@agentic.logging.log_call(logger)
def query_database(sql: str) -> list[dict[str, Any]]:
    """Run a read-only SELECT query against the customer database and return the results.

    Only SELECT statements are permitted; any other statement is rejected.

    Args:
        sql: A SQL SELECT query to execute.

    Returns:
        A list of rows, each represented as a dictionary of column names to values.
    """
    if not sql.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed.")

    conn = _connect()
    try:
        cursor = conn.cursor()
        cursor.execute("SET TRANSACTION READ ONLY")
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
    finally:
        conn.close()

    return [dict(zip(columns, row, strict=True)) for row in rows]


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------


def main(
    transport: Annotated[Literal["stdio", "http"], typer.Argument()] = "stdio",
) -> None:
    """Model Context Protocol (MCP) Customer Database Server."""
    logger.info(f"Starting {transport} MCP Customer Database Server")

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
