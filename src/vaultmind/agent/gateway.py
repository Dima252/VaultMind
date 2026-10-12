"""The tool gateway: every tool call the agent makes goes through here.

It validates arguments, enforces the step and observation budgets (`agent.max_steps`,
`agent.observation_tokens`), answers a repeated call from its first result, asks for approval
before a write, and runs every observation through the injection guard and wraps it as data.
"""

from __future__ import annotations

from collections.abc import Sequence

from vaultmind.agent.tools import Tool, ToolContext
from vaultmind.agent.types import ToolResult
from vaultmind.core.providers import ToolCall, ToolSpec


class Gateway:
    """One per agent run (the step budget and the repeat cache are per run)."""

    def __init__(self, ctx: ToolContext, tools: Sequence[Tool] | None = None) -> None:
        """`tools` defaults to `default_tools()`."""
        raise NotImplementedError

    def specs(self) -> list[ToolSpec]:
        raise NotImplementedError

    @property
    def steps(self) -> int:
        """Tool calls executed so far (repeats answered from cache don't count)."""
        raise NotImplementedError

    def call(self, call: ToolCall, *, approved: bool = False) -> ToolResult:
        """Run one call. Problems the model can fix (unknown tool, invalid arguments,
        `ToolError` such as `DocumentNotFound`, the step budget used up) come back as a
        failed `ToolResult` whose `error` says what to change. Raises `AmbiguousDocument`
        and, for a write tool without `approved`, `ApprovalRequired`: both need the user."""
        raise NotImplementedError
