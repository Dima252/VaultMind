"""The "localhost only" privacy badge: check this process's network connections."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class PrivacyStatus:
    local_only: bool
    remote: tuple[str, ...]  # "<ip>:<port> (<process name>)" per non-loopback connection
    checked_at: datetime  # UTC


def is_loopback(ip: str) -> bool:
    """127.0.0.0/8, ::1 and IPv4-mapped loopback (::ffff:127.0.0.1)."""
    raise NotImplementedError


def check(pid: int | None = None) -> PrivacyStatus:
    """Connections of process `pid` (default: this one) and all its children, e.g. the
    webview's browser processes. Listening sockets and sockets without a remote end are fine."""
    raise NotImplementedError
