"""Only VaultMind's own page may talk to its server.

Binding to 127.0.0.1 keeps the network out, but not other websites open in a browser on this
machine. A page can reach http://127.0.0.1:<port> through DNS rebinding (its own domain
resolving to 127.0.0.1, so the Host header is that domain) or open a cross-origin socket
(NiceGUI's socket.io accepts any origin). Either would let it read chats and drive the agent.
So every HTTP and websocket request must name our address as Host, and as Origin when a
browser sends one.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


class LocalOnly:
    """ASGI middleware: 403 for a foreign Host or Origin."""

    def __init__(self, app: ASGIApp, *, host: str, port: int) -> None:
        self.app = app
        self.hosts = {f"{host}:{port}", f"localhost:{port}"}
        self.origins = {f"http://{h}" for h in self.hosts}

    def allowed(self, headers: dict[str, str]) -> bool:
        origin = headers.get("origin")
        return headers.get("host") in self.hosts and (origin is None or origin in self.origins)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        if self.allowed(headers):
            await self.app(scope, receive, send)
        elif scope["type"] == "websocket":
            await receive()  # websocket.connect
            await send({"type": "websocket.close", "code": 1008})  # before accept: HTTP 403
        else:
            body = b"Forbidden: VaultMind only answers its own page."
            await send(
                {
                    "type": "http.response.start",
                    "status": 403,
                    "headers": [
                        (b"content-type", b"text/plain"),
                        (b"content-length", b"%d" % len(body)),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
