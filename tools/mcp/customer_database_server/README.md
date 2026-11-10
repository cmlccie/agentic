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

### Caller Authorization

Off by default: any caller that can reach the server can call every tool. `AUTH_TOOL_ROLES`
turns it on **per tool call** for the HTTP transport (stdio is never checked). It is a JSON
object mapping a tool name to the role its callers must hold, for example
`{"query_database": "query"}`.

- Connecting, listing tools, and calling a tool that is not in the map need no credential,
  so a gateway's health check and tool discovery keep working.
- A call to a mapped tool must arrive on a request carrying `Authorization: Bearer <access
  token>`. The token must be a JWT signed by a key published at `AUTH_JWKS_URI`, with `iss`
  equal to `AUTH_ISSUER` and `aud` containing `AUTH_AUDIENCE`, and it must hold the tool's
  role as a Keycloak client role on the audience client
  (`resource_access.<AUTH_AUDIENCE>.roles`).
- The token is read from the request that carries the call, not from the one that opened
  the session.
- A refused call is returned as a tool error (`isError: true`), not as an HTTP `401`.

The server fails at startup if `AUTH_TOOL_ROLES` is set and any of the other three is not,
or if it names a tool that does not exist.

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
- `AUTH_TOOL_ROLES`: Unset by default. JSON object mapping a tool name to the role required
  to call it; setting it turns caller authorization on and makes the next three required.
- `AUTH_JWKS_URI`: JWKS endpoint of the token issuer.
- `AUTH_ISSUER`: Expected `iss` claim.
- `AUTH_AUDIENCE`: Expected `aud` claim, and the Keycloak client whose roles are checked.
- `HOST`: `0.0.0.0` by default. HTTP server bind host.
- `PORT`: `8000` by default. HTTP server bind port.

## Build Locally

```bash
# Build for your current platform
docker build -t tools-mcp-customer-database-server .

# Build for multiple platforms
docker buildx build --platform linux/amd64,linux/arm64 -t agentic/tools-mcp-customer-database-server:local .
```
