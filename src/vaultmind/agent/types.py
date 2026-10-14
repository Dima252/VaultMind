"""Results and trace events of the agent (Phase 4), plus the document-agent types (Phase 5.5).

Shared contract between the graph (`agent/graph.py`), the safety checks (`agent/checks.py`),
the tools and gateway (`agent/tools.py`, `agent/gateway.py`), the trace store
(`store/traces.py`), the eval runner and the CLI. Change it deliberately.

Trace events: every graph node appends one `TraceEvent` per execution, in order (the recipe
node also appends one `tool` event per tool call, before its own). `data` is JSON-serializable
and uses these keys (a node may add more; consumers must tolerate extras):

| node             | data keys                                                          |
|------------------|--------------------------------------------------------------------|
| `analyze`        | `standalone_query` str, `variants` list[str], `needs_retrieval` bool, |
|                  | `skipped` bool (only when no LLM call was needed)                  |
| `retrieve`       | `queries` list[str], `candidates` int                              |
| `grade`          | `threshold` float or None, `passed` int,                           |
|                  | `scores` list of {`label` str, `score` float, `passed` bool},      |
|                  | `flagged` list of {`label` str, `warnings` list[str]}              |
| `rewrite`        | `attempt` int (1-based), `query` str, `repeated` bool (only if so) |
| `contradictions` | `found` list of {`claim_a`, `source_a`, `claim_b`, `source_b`}     |
| `generate`       | `sources` list[int] (numbers shown), `abstained` bool              |
| `not_found`      | `closest` list[str] (source labels), `missing` list[str] (only     |
|                  | when the planner named documents that aren't there)                |
| `groundedness`   | `unsupported` list of {`claim`, `cited`, `reason`}                 |
| `direct`         | `reason` str (answered without retrieval), `closest` list[str]     |
|                  | (only when it searched and abstained)                              |
| `plan`           | `task` str, `mentions` list[str], `documents` list[str] (file ids), |
|                  | `language` str, `steps` list[str], `clarify` str or None,          |
|                  | `missing` list[str] (mentions that matched no document),           |
|                  | `route` str (the node it leads to); `focus` str and `fields`       |
|                  | list[str] only when set                                            |
| `recipe`         | `name` str (the task), `documents` list[str] (file ids)            |
| `reason`         | `step` int (1-based), `tool_calls` list[str] (names),              |
|                  | `thinking_chars` int                                               |
| `tool`           | `name` str, `arguments` dict, `ok` bool, `error` str or None,      |
|                  | `chars` int (observation length), `sources` int, `warnings` list[str] |
| `human`          | `kind` "clarify", "confirm" or "approve", `prompt` str,            |
|                  | `decision` str (the user's choice or reply)                        |
| `respond`        | `task` str, `sources` list[int] (numbers shown)                    |

A node that fails without stopping the run (e.g. an optional check whose LLM output can't be
parsed) records `error` str in its `data`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from vaultmind.agent.baseline import Source, cited_sources
from vaultmind.core.providers import ToolCall

NODES: tuple[str, ...] = (
    "analyze",
    "retrieve",
    "grade",
    "rewrite",
    "contradictions",
    "generate",
    "not_found",
    "groundedness",
    "direct",
    "plan",
    "recipe",
    "reason",
    "tool",
    "human",
    "respond",
)

TaskKind = Literal[
    "chat",  # greeting, thanks, a question about the assistant
    "transform",  # rework the previous answer: shorter, translated, as a table (no retrieval)
    "lookup",  # a fact from the documents: the corrective RAG graph
    "list",  # what documents there are, or which ones match a filter
    "summarize",  # one or more whole documents
    "compare",  # two or more documents side by side
    "extract",  # the same fields from several documents, answered as a table
    "generate",  # study notes, a quiz, a draft, made from documents
    "open",  # anything multi-step the above don't cover: the tool loop
]
TASKS: tuple[TaskKind, ...] = (
    "chat",
    "transform",
    "lookup",
    "list",
    "summarize",
    "compare",
    "extract",
    "generate",
    "open",
)


@dataclass(frozen=True)
class TraceEvent:
    node: str  # one of NODES
    duration_ms: float
    data: dict[str, Any] = field(default_factory=dict)  # JSON-serializable; keys per node above


@dataclass(frozen=True)
class Plan:
    """What the planner made of a message."""

    task: TaskKind
    mentions: tuple[str, ...] = ()  # documents as the user named them ("the syllabus")
    documents: tuple[str, ...] = ()  # the file ids they resolved to, in the order named
    language: str = "en"  # ISO 639-1 to answer in
    steps: tuple[str, ...] = ()  # 1-3 short steps, shown to the user while it works
    clarify: str | None = None  # ask the user this instead of acting
    focus: str | None = None  # a topic the request narrows to ("grading", "termination")
    fields: tuple[str, ...] = ()  # extract: what to pull out of each document
    calculate: bool = False  # the answer needs arithmetic or date counting (any task)


@dataclass(frozen=True)
class ToolResult:
    """One executed tool call, as the gateway returns it."""

    call: ToolCall
    ok: bool
    content: str  # the observation shown to the model: budgeted, guarded, wrapped as data
    sources: tuple[Source, ...] = ()  # text the call read, citable; `n` is 0 until numbered
    warnings: tuple[str, ...] = ()  # injection-guard flags on the observation
    error: str | None = None  # why it failed: bad arguments, unknown document, budget
    duration_ms: float = 0.0


@dataclass(frozen=True)
class Contradiction:
    """Two graded passages (by source number) that disagree on a fact the question asks about."""

    claim_a: str
    source_a: int
    claim_b: str
    source_b: int


@dataclass(frozen=True)
class UnsupportedClaim:
    """An answer claim that its cited passages don't support."""

    claim: str
    cited: tuple[int, ...]  # source numbers the answer cited for it (may be empty)
    reason: str


@dataclass(frozen=True)
class AgentResult:
    question: str
    answer: str
    sources: list[Source]  # passages that passed grading; on abstention, the closest matches
    abstained: bool
    queries: list[str]  # every query searched, in order; the standalone question comes first
    rewrites: int  # rewrite rounds used (0 = the first search was enough)
    contradictions: list[Contradiction] = field(default_factory=list)
    unsupported: list[UnsupportedClaim] = field(default_factory=list)  # only with the check on
    trace: list[TraceEvent] = field(default_factory=list)

    @property
    def cited(self) -> list[Source]:
        return cited_sources(self.answer, self.sources)

    @property
    def task(self) -> str:
        """The route the run took: the planner's task, or "lookup" when no planner ran."""
        for event in self.trace:
            if event.node == "plan":
                return str(event.data.get("task", "lookup"))
        return "lookup"

    def node_ms(self, node: str) -> float:
        """Total time spent in `node` across the run."""
        return sum(e.duration_ms for e in self.trace if e.node == node)


class ToolError(Exception):
    """A tool call failed in a way the model can fix; the message becomes the observation."""


class DocumentNotFound(ToolError):
    """A document the request or a tool call named matches no file in the vault."""

    def __init__(self, mention: str) -> None:
        super().__init__(f"No document matches {mention!r}.")
        self.mention = mention


class AmbiguousDocument(Exception):
    """A mention matches several documents: the agent asks the user which one (clarify)."""

    def __init__(self, mention: str, candidates: tuple[str, ...]) -> None:
        super().__init__(f"{mention!r} could be any of: {', '.join(candidates)}")
        self.mention = mention
        self.candidates = candidates  # vault-relative file names


class ApprovalRequired(Exception):
    """A write tool was called without the user's approval: the agent asks for it."""

    def __init__(self, call: ToolCall) -> None:
        super().__init__(f"{call.name} needs your approval")
        self.call = call
