"""Authentication dependencies (REQ-SEC-04, REQ-SEC-06; docs/API.md §1).

REST endpoints use the `Authorization` header; the WebSocket upgrade uses a query-string token
instead, because browsers cannot set custom headers on a WS handshake. Same secret (`API_KEY`),
different transport — REST auth is unchanged by the WS variant.
"""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.core.config import get_settings


async def require_api_key(authorization: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if authorization != f"Bearer {settings.api_key}":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing API key"
        )


async def require_eval_key(
    x_eval_key: str | None = Header(default=None, alias="X-Eval-Key"),
) -> None:
    settings = get_settings()
    if x_eval_key != settings.eval_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="invalid or missing eval key"
        )


def ws_token_is_valid(token: str | None) -> bool:
    return token == get_settings().api_key
