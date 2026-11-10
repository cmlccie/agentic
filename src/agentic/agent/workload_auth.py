"""SPIFFE workload identity for calls to the AI gateway.

Instead of a static API key, the agent proves who it is with a short-lived,
attested identity:

1. Fetch a JWT-SVID from the SPIFFE Workload API (the SPIRE agent's socket).
2. Exchange it at the identity provider's token endpoint as a federated client
   assertion (OAuth 2.0 client credentials, SPIFFE flavor) for an access token.
3. Send the access token as ``Authorization: Bearer`` on every gateway request,
   refreshing it shortly before it expires.

Enabled entirely by environment variables, and inert when they are unset:

- ``AGENT_WORKLOAD_AUTH_TOKEN_URL`` — the token endpoint. Setting it turns
  workload auth on.
- ``AGENT_WORKLOAD_AUTH_AUDIENCE`` — the audience requested in the JWT-SVID. For
  Keycloak this is the realm's issuer URL.
- ``AGENT_WORKLOAD_AUTH_BASE_URL`` — the gateway's base URL. Only requests to
  this origin carry the token, so it is never sent to any other host.
- ``SPIFFE_ENDPOINT_SOCKET`` — the Workload API socket, read by the ``spiffe``
  library itself (``unix:///path/to/spire-agent.sock``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncGenerator, Callable, Generator
from dataclasses import dataclass
from functools import cache
from urllib.parse import urlsplit

import httpx2

from .config import ConfigError

log = logging.getLogger(__name__)

TOKEN_URL_ENV = "AGENT_WORKLOAD_AUTH_TOKEN_URL"
AUDIENCE_ENV = "AGENT_WORKLOAD_AUTH_AUDIENCE"
BASE_URL_ENV = "AGENT_WORKLOAD_AUTH_BASE_URL"

#: Client assertion type for a JWT-SVID (OAuth SPIFFE Client Authentication draft).
CLIENT_ASSERTION_TYPE = "urn:ietf:params:oauth:client-assertion-type:jwt-spiffe"

#: Refresh the access token this long before it expires.
REFRESH_MARGIN_SECONDS = 30.0

#: Placeholder API key for OpenAI-compatible clients, which require one. The
#: `Authorization` header it produces is replaced by the access token.
PLACEHOLDER_API_KEY = "workload-identity"


@dataclass(frozen=True)
class WorkloadAuthSettings:
    """Where to exchange the JWT-SVID, and which origin receives the token."""

    token_url: str
    audience: str
    base_url: str


@dataclass(frozen=True)
class AccessToken:
    """An access token and the `time.monotonic()` value it expires at."""

    value: str
    expires_at: float


def load_settings() -> WorkloadAuthSettings | None:
    """Read the settings from the environment, or None when workload auth is off.

    Raises:
        ConfigError: the token URL is set but another required variable is not.
    """
    token_url = os.environ.get(TOKEN_URL_ENV)
    if not token_url:
        return None
    values = {name: os.environ.get(name) for name in (AUDIENCE_ENV, BASE_URL_ENV)}
    if missing := sorted(name for name, value in values.items() if not value):
        raise ConfigError(f"{TOKEN_URL_ENV} is set, so {missing} must be set as well")
    return WorkloadAuthSettings(
        token_url=token_url,
        audience=values[AUDIENCE_ENV] or "",
        base_url=values[BASE_URL_ENV] or "",
    )


def origin(url: str) -> tuple[str, str, int | None]:
    """The (scheme, host, port) of a URL, with default ports made explicit."""
    parts = urlsplit(url)
    port = parts.port or {"http": 80, "https": 443}.get(parts.scheme)
    return parts.scheme, (parts.hostname or "").lower(), port


def fetch_jwt_svid(audience: str) -> str:
    """Fetch a JWT-SVID for `audience` from the SPIFFE Workload API (blocking).

    `spiffe` is imported here so that agents without workload auth never load it
    or its gRPC dependency.
    """
    from spiffe import WorkloadApiClient

    client = WorkloadApiClient()
    try:
        return client.fetch_jwt_svid(audience={audience}).token
    finally:
        client.close()


async def exchange_svid(
    client: httpx2.AsyncClient, token_url: str, svid: str
) -> AccessToken:
    """Exchange a JWT-SVID for an access token at the token endpoint."""
    requested_at = time.monotonic()
    response = await client.post(
        token_url,
        data={
            "grant_type": "client_credentials",
            "client_assertion_type": CLIENT_ASSERTION_TYPE,
            "client_assertion": svid,
        },
    )
    if response.is_error:
        raise httpx2.HTTPStatusError(
            f"Token exchange at {token_url} failed with HTTP {response.status_code}: "
            f"{response.text[:500]}",
            request=response.request,
            response=response,
        )
    body = response.json()
    return AccessToken(
        value=body["access_token"],
        expires_at=requested_at + float(body["expires_in"]),
    )


class WorkloadAuth(httpx2.Auth):
    """`httpx2` auth that sends a workload-identity access token as the bearer.

    The token is cached and shared by every client using this instance, and is
    replaced once, with a fresh one, if the gateway answers 401.
    """

    def __init__(
        self,
        settings: WorkloadAuthSettings,
        *,
        fetch_svid: Callable[[str], str] = fetch_jwt_svid,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        """
        Args:
            settings: Token endpoint, SVID audience, and gateway base URL.
            fetch_svid: Blocking callable returning a JWT-SVID for an audience.
            transport: Transport for the token-endpoint client (for tests).
        """
        self.settings = settings
        self._fetch_svid = fetch_svid
        self._transport = transport
        self._token: AccessToken | None = None
        self._lock = asyncio.Lock()

    async def access_token(self, *, force: bool = False) -> str:
        """Return a valid access token, exchanging a fresh JWT-SVID if needed."""
        async with self._lock:
            token = self._token
            fresh_until = time.monotonic() + REFRESH_MARGIN_SECONDS
            if force or token is None or token.expires_at <= fresh_until:
                token = self._token = await self._obtain()
            return token.value

    async def _obtain(self) -> AccessToken:
        # py-spiffe is synchronous; keep the event loop free while it talks gRPC.
        svid = await asyncio.to_thread(self._fetch_svid, self.settings.audience)
        async with httpx2.AsyncClient(transport=self._transport) as client:
            token = await exchange_svid(client, self.settings.token_url, svid)
        log.debug("Obtained a workload-identity access token")
        return token

    async def async_auth_flow(
        self, request: httpx2.Request
    ) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
        request.headers["Authorization"] = f"Bearer {await self.access_token()}"
        response = yield request
        if response.status_code == 401:
            log.warning("Gateway rejected the access token; retrying with a new one")
            token = await self.access_token(force=True)
            request.headers["Authorization"] = f"Bearer {token}"
            yield request

    def sync_auth_flow(
        self, request: httpx2.Request
    ) -> Generator[httpx2.Request, httpx2.Response]:
        raise RuntimeError("WorkloadAuth supports only asynchronous HTTP clients")


@cache
def _shared_auth(settings: WorkloadAuthSettings) -> WorkloadAuth:
    return WorkloadAuth(settings)


def auth_for(url: str) -> WorkloadAuth | None:
    """Return the shared `WorkloadAuth` for a gateway URL, or None.

    None when workload auth is off, or when `url` is not on the gateway's origin
    — the access token must never be sent anywhere else. One instance is shared
    per process so model and MCP calls reuse the same token.
    """
    settings = load_settings()
    if settings is None or origin(url) != origin(settings.base_url):
        return None
    return _shared_auth(settings)
