"""Header privacy badge: "localhost only", or red with the remote addresses it saw."""

from __future__ import annotations


def privacy_badge() -> None:
    """Build the badge; it re-checks every `ui.privacy_check_s` seconds (`app/privacy.py`)."""
