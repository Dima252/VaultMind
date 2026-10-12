"""The privacy badge: which of the app's connections leave the machine."""

from dataclasses import dataclass, field
from datetime import UTC
from types import SimpleNamespace
from typing import Any

import psutil
import pytest

from vaultmind.app.privacy import PrivacyStatus, check, is_loopback


@pytest.mark.parametrize(
    ("ip", "expected"),
    [
        ("127.0.0.1", True),
        ("127.255.0.9", True),
        ("::1", True),
        ("::ffff:127.0.0.1", True),
        ("[::1]", True),
        ("0:0:0:0:0:0:0:1", True),
        ("192.168.1.10", False),
        ("10.0.0.1", False),
        ("8.8.8.8", False),
        ("::ffff:8.8.8.8", False),
        ("2001:db8::1", False),
        ("fe80::1%3", False),  # scoped link-local: not loopback, and no crash
        ("::1%1", True),
        ("0.0.0.0", False),
        ("localhost", False),  # not an address
        ("", False),
        ("not an ip", False),
    ],
)
def test_is_loopback(ip: str, expected: bool) -> None:
    assert is_loopback(ip) is expected


def _conn(ip: str | None, port: int = 443) -> SimpleNamespace:
    """psutil reports listening sockets with an empty `raddr`."""
    return SimpleNamespace(raddr=SimpleNamespace(ip=ip, port=port) if ip else ())


@dataclass
class FakeProcess:
    pname: str
    conns: list[SimpleNamespace] = field(default_factory=list)
    kids: list[Any] = field(default_factory=list)
    error: Exception | None = None
    kinds: list[str] = field(default_factory=list)

    def name(self) -> str:
        if self.error is not None:
            raise self.error
        return self.pname

    def net_connections(self, kind: str) -> list[SimpleNamespace]:
        self.kinds.append(kind)
        return self.conns

    def children(self, recursive: bool) -> list[Any]:
        assert recursive
        return self.kids


def _install(monkeypatch: pytest.MonkeyPatch, root: Any) -> list[int | None]:
    pids: list[int | None] = []

    def process(pid: int | None = None) -> Any:
        pids.append(pid)
        return root

    monkeypatch.setattr(psutil, "Process", process)
    return pids


def test_only_remote_connections_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    webview = FakeProcess(
        "msedgewebview2.exe", [_conn("93.184.216.34", 443), _conn("::ffff:127.0.0.1", 8730)]
    )
    vanished = FakeProcess("gone.exe", error=psutil.NoSuchProcess(4242))
    denied = FakeProcess("system.exe", error=psutil.AccessDenied(4))
    root = FakeProcess(
        "python.exe",
        [
            _conn("127.0.0.1", 11434),  # Ollama
            _conn("::1", 8730),
            _conn(None),  # listening
            _conn("2001:db8::7", 80),
            _conn("93.184.216.34", 443),
        ],
        kids=[webview, vanished, denied],
    )
    pids = _install(monkeypatch, root)

    status = check(1234)

    assert pids == [1234]
    assert isinstance(status, PrivacyStatus)
    assert not status.local_only
    assert status.remote == (
        "93.184.216.34:443 (msedgewebview2.exe)",
        "93.184.216.34:443 (python.exe)",
        "[2001:db8::7]:80 (python.exe)",
    )
    assert status.checked_at.tzinfo is UTC
    assert root.kinds == ["inet"]


def test_loopback_only_is_local(monkeypatch: pytest.MonkeyPatch) -> None:
    root = FakeProcess("python.exe", [_conn("127.0.0.1", 11434), _conn(None)])
    _install(monkeypatch, root)
    status = check()
    assert status.local_only
    assert status.remote == ()


def test_duplicates_are_reported_once(monkeypatch: pytest.MonkeyPatch) -> None:
    root = FakeProcess("python.exe", [_conn("8.8.8.8", 53), _conn("8.8.8.8", 53)])
    _install(monkeypatch, root)
    assert check().remote == ("8.8.8.8:53 (python.exe)",)


def test_children_that_cannot_be_listed_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    class NoChildren(FakeProcess):
        def children(self, recursive: bool) -> list[Any]:
            raise psutil.AccessDenied(1)

    _install(monkeypatch, NoChildren("python.exe", [_conn("1.1.1.1", 443)]))
    assert check().remote == ("1.1.1.1:443 (python.exe)",)


def test_older_psutil_connections_method(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class OldProcess:
        def name(self) -> str:
            return "python.exe"

        def connections(self, kind: str) -> list[SimpleNamespace]:
            calls.append(kind)
            return [_conn("9.9.9.9", 853)]

        def children(self, recursive: bool) -> list[Any]:
            return []

    _install(monkeypatch, OldProcess())
    assert check().remote == ("9.9.9.9:853 (python.exe)",)
    assert calls == ["inet"]


def test_real_process() -> None:
    status = check()
    assert isinstance(status, PrivacyStatus)
    assert status.local_only == (status.remote == ())
