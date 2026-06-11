from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..lifespan import AppState


class _DynamicUIApp:
    """ASGI wrapper that lazily creates the Pydantic AI web UI and
    recreates it when the agent is replaced on reload."""

    def __init__(self, app_state: AppState) -> None:
        self._state = app_state
        self._last_agent: Any = None
        self._web_app: Any = None

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        agent = self._state.agent
        if agent is not self._last_agent:
            self._last_agent = agent
            self._web_app = agent.to_web()
        await self._web_app(scope, receive, send)


def build_ui_app(app_state: AppState) -> _DynamicUIApp:
    return _DynamicUIApp(app_state)
