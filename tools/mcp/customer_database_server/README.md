# MCP Customer Database Server

A Model Context Protocol (MCP) server that provides read-only SQL access to a PostgreSQL
customer database (customers, products, and purchases).

## Tools

- `get_schema`: Return a DDL-like description of all tables and columns in the database.
- `query_database`: Run a read-only query and return `{rows, truncated}`. Statements must
  start with `SELECT`, `WITH`, or `EXPLAIN` (leading comments are ignored), run inside a
  `READ ONLY` transaction with a statement timeout, and return at most `QUERY_MAX_ROWS`
  rows.

## Security

The statement-prefix check and the `READ ONLY` transaction are defense in depth, not the
access control. The real control is the database role: connect as a role that has only
`SELECT` privileges on the tables the agent may read, for example:

```sql
CREATE ROLE customers_reader LOGIN PASSWORD '...';
GRANT CONNECT ON DATABASE customers TO customers_reader;
GRANT USAGE ON SCHEMA public TO customers_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO customers_reader;
```

Query results are not logged, because they may contain personal data.

## Container Image

This project is automatically built and published as a multi-architecture container image supporting both `amd64` and `arm64` platforms.

### Available Images

- `ghcr.io/cmlccie/agentic/tools-mcp-customer-database-server:latest`: Latest version from main branch
- `ghcr.io/cmlccie/agentic/tools-mcp-customer-database-server:main`: Main branch builds
- `ghcr.io/cmlccie/agentic/tools-mcp-customer-database-server:v*`: Tagged releases

### Use

```bash
# Run in stdio mode (default)
docker run --rm ghcr.io/cmlccie/agentic/tools-mcp-customer-database-server:latest

# Run in HTTP mode on port 8000
docker run --rm -p 8000:8000 \
  -e PGHOST=customer-database -e PGDATABASE=customers \
  -e PGUSER=customers -e PGPASSWORD=... \
  ghcr.io/cmlccie/agentic/tools-mcp-customer-database-server:latest http
```

The HTTP MCP endpoint is exposed at `/mcp`.

## Configuration

### Required Environment Variables

- `PGHOST`: PostgreSQL host.
- `PGDATABASE`: Database name.
- `PGUSER`: Database user.
- `PGPASSWORD`: Database password.

### Optional Environment Variables

- `PGPORT`: `5432` by default. PostgreSQL port.
- `QUERY_TIMEOUT_S`: `5` by default. Per-query statement timeout in seconds.
- `QUERY_MAX_ROWS`: `500` by default. Maximum rows returned by `query_database`.
- `HOST`: `0.0.0.0` by default. HTTP server bind host.
- `PORT`: `8000` by default. HTTP server bind port.

## Build Locally

```bash
# Build for your current platform
docker build -t tools-mcp-customer-database-server .

# Build for multiple platforms
docker buildx build --platform linux/amd64,linux/arm64 -t agentic/tools-mcp-customer-database-server:local .
```
