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

    def fetchall(self):
        return self.rows


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


# -------------------------------------------------------------------------------------------------
# Caller authentication and authorization
# -------------------------------------------------------------------------------------------------

ISSUER = "https://keycloak.example/realms/lab"
AUDIENCE = "customer-database-mcp-server"


@pytest.fixture
def auth(monkeypatch):
    """The server module with ``query_database`` requiring the ``query`` role, plus a signing key."""
    from fastmcp.server.auth.providers.jwt import JWTVerifier, RSAKeyPair

    monkeypatch.setenv("AUTH_TOOL_ROLES", '{"query_database": "query"}')
    monkeypatch.setenv("AUTH_JWKS_URI", "http://keycloak.example/certs")
    monkeypatch.setenv("AUTH_ISSUER", ISSUER)
    monkeypatch.setenv("AUTH_AUDIENCE", AUDIENCE)
    module = load_server("customer_database_server")

    # Same issuer and audience as the server's verifier, but with a local key
    # instead of a JWKS fetch.
    key = RSAKeyPair.generate()
    (middleware,) = (
        m for m in module.mcp.middleware if isinstance(m, module.ToolRoleMiddleware)
    )
    middleware.verifier = JWTVerifier(
        public_key=key.public_key, issuer=ISSUER, audience=AUDIENCE
    )
    monkeypatch.setattr(module, "_connect", lambda: FakeConnection(FakeCursor([])))
    return module, key


def _token(key, *, audience=AUDIENCE, roles=("query",), roles_on=AUDIENCE):
    return key.create_token(
        subject="service-account-customer-database-agent",
        issuer=ISSUER,
        audience=audience,
        additional_claims={"resource_access": {roles_on: {"roles": list(roles)}}},
    )


def _present(monkeypatch, module, token):
    """Make ``token`` the bearer on the request carrying the next tool calls."""
    monkeypatch.setattr(module, "_bearer_token", lambda: token)


def test_auth_is_off_by_default(db):
    assert db.build_middleware() == []


def test_auth_requires_every_setting(monkeypatch):
    monkeypatch.setenv("AUTH_TOOL_ROLES", '{"query_database": "query"}')
    monkeypatch.setenv("AUTH_ISSUER", ISSUER)
    with pytest.raises(ValueError, match="AUTH_AUDIENCE.*AUTH_JWKS_URI"):
        load_server("customer_database_server")


def test_auth_rejects_unknown_tool_names(monkeypatch):
    monkeypatch.setenv("AUTH_TOOL_ROLES", '{"query_databse": "query"}')
    monkeypatch.setenv("AUTH_JWKS_URI", "http://keycloak.example/certs")
    monkeypatch.setenv("AUTH_ISSUER", ISSUER)
    monkeypatch.setenv("AUTH_AUDIENCE", AUDIENCE)
    with pytest.raises(ValueError, match="unknown tools.*query_databse"):
        load_server("customer_database_server")


def test_unauthenticated_caller_lists_and_calls_open_tools(auth):
    module, _ = auth
    assert list_tool_names(module.mcp) == {"get_schema", "query_database"}
    call_tool(module.mcp, "get_schema")


def test_unauthenticated_caller_is_refused_a_protected_tool(auth):
    module, _ = auth
    with pytest.raises(ToolError, match="requires a valid access token"):
        call_tool(module.mcp, "query_database", {"sql": "SELECT 1"})


def test_invalid_token_still_calls_open_tools(auth, monkeypatch):
    module, _ = auth
    _present(monkeypatch, module, "not-a-jwt")
    call_tool(module.mcp, "get_schema")


def test_token_with_the_role_calls_a_protected_tool(auth, monkeypatch):
    module, key = auth
    _present(monkeypatch, module, _token(key))
    call_tool(module.mcp, "query_database", {"sql": "SELECT 1"})


@pytest.mark.parametrize(
    ("claims", "error"),
    [
        ({"roles": ()}, "requires the 'query' role"),
        ({"roles": ("other",)}, "requires the 'query' role"),
        # the role on another client does not count
        ({"roles_on": "litellm"}, "requires the 'query' role"),
        # a gateway token is not a token for this server
        ({"audience": "litellm"}, "requires a valid access token"),
    ],
)
def test_other_tokens_are_refused_a_protected_tool(auth, monkeypatch, claims, error):
    module, key = auth
    _present(monkeypatch, module, _token(key, **claims))
    with pytest.raises(ToolError, match=error):
        call_tool(module.mcp, "query_database", {"sql": "SELECT 1"})
