"""The "localhost only" privacy badge: check this process's network connections."""

from __future__ import annotations

import ipaddress
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import psutil  # type: ignore[import-untyped]

_GONE = (psutil.NoSuchProcess, psutil.AccessDenied)  # ZombieProcess is a NoSuchProcess


@dataclass(frozen=True)
class PrivacyStatus:
    local_only: bool
    remote: tuple[str, ...]  # "<ip>:<port> (<process name>)" per non-loopback connection
    checked_at: datetime  # UTC


def is_loopback(ip: str) -> bool:
    """127.0.0.0/8, ::1 and IPv4-mapped loopback (::ffff:127.0.0.1)."""
    try:
        addr = ipaddress.ip_address(ip.strip().strip("[]").split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return addr.ipv4_mapped.is_loopback
    return addr.is_loopback


def check(pid: int | None = None) -> PrivacyStatus:
    """Connections of process `pid` (default: this one) and all its children, e.g. the
    webview's browser processes. Listening sockets and sockets without a remote end are fine."""
    remote: set[str] = set()
    for proc in _processes(pid):
        try:
            name = proc.name()
            conns = _connections(proc)
        except _GONE:
            continue
        for conn in conns:
            raddr = conn.raddr
            if not raddr or is_loopback(raddr.ip):
                continue
            host = f"[{raddr.ip}]" if ":" in raddr.ip else raddr.ip
            remote.add(f"{host}:{raddr.port} ({name})")
    return PrivacyStatus(
        local_only=not remote, remote=tuple(sorted(remote)), checked_at=datetime.now(UTC)
    )


def _processes(pid: int | None) -> Iterator[Any]:
    root = psutil.Process(pid)
    yield root
    try:
        children = root.children(recursive=True)
    except _GONE:
        return
    yield from children


def _connections(proc: Any) -> list[Any]:
    get = getattr(proc, "net_connections", None) or proc.connections  # psutil < 6
    return list(get(kind="inet"))
