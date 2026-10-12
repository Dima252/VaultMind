"""The UI server answers only its own page: no foreign Host (DNS rebinding), no foreign Origin."""

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route, WebSocketRoute
from starlette.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect

from vaultmind.ui.local_only import LocalOnly

PORT = 8730
OURS = f"127.0.0.1:{PORT}"


async def _page(_: object) -> PlainTextResponse:
    return PlainTextResponse("your chats")


async def _socket(ws: WebSocket) -> None:
    await ws.accept()
    await ws.send_text("hello")
    await ws.close()


@pytest.fixture
def client() -> TestClient:
    inner = Starlette(routes=[Route("/", _page), WebSocketRoute("/ws", _socket)])
    return TestClient(LocalOnly(inner, host="127.0.0.1", port=PORT))


@pytest.mark.parametrize("host", [OURS, f"localhost:{PORT}"])
def test_own_page_is_served(client: TestClient, host: str) -> None:
    response = client.get("/", headers={"host": host, "origin": f"http://{host}"})
    assert response.status_code == 200
    assert client.get("/", headers={"host": host}).status_code == 200  # no Origin: navigation


@pytest.mark.parametrize(
    "headers",
    [
        {"host": f"evil.example:{PORT}"},  # DNS rebinding
        {"host": "127.0.0.1:9999"},
        {"host": OURS, "origin": "https://evil.example"},  # another site's script
        {"host": OURS, "origin": "null"},
    ],
)
def test_foreign_requests_are_refused(client: TestClient, headers: dict[str, str]) -> None:
    response = client.get("/", headers=headers)
    assert response.status_code == 403
    assert "your chats" not in response.text


def test_websocket_from_own_page_connects(client: TestClient) -> None:
    with client.websocket_connect("/ws", headers={"host": OURS, "origin": f"http://{OURS}"}) as ws:
        assert ws.receive_text() == "hello"


def test_websocket_from_another_site_is_refused(client: TestClient) -> None:
    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect("/ws", headers={"host": OURS, "origin": "https://evil.example"}),
    ):
        pass
