"""Results and trace events of the corrective agent (Phase 4).

Shared contract between the graph (`agent/graph.py`), the safety checks (`agent/checks.py`),
the trace store (`store/traces.py`), the eval runner and the CLI. Change it deliberately.

Trace events: every graph node appends one `TraceEvent` per execution, in order. `data` is
JSON-serializable and uses these keys (a node may add more; consumers must tolerate extras):

| node             | data keys                                                          |
|------------------|--------------------------------------------------------------------|
| `analyze`        | `standalone_query` str, `variants` list[str], `needs_retrieval` bool |
| `retrieve`       | `queries` list[str], `candidates` int                              |
| `grade`          | `threshold` float or None, `passed` int,                           |
|                  | `scores` list of {`label` str, `score` float, `passed` bool}        |
| `rewrite`        | `attempt` int (1-based), `query` str                               |
| `contradictions` | `found` list of {`claim_a`, `source_a`, `claim_b`, `source_b`}     |
| `generate`       | `sources` list[int] (numbers shown), `abstained` bool              |
| `not_found`      | `closest` list[str] (source labels)                                |
| `groundedness`   | `unsupported` list of {`claim`, `cited`, `reason`}                 |
| `direct`         | `reason` str (answered without retrieval)                          |

A node that fails without stopping the run (e.g. an optional check whose LLM output can't be
parsed) records `error` str in its `data`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vaultmind.agent.baseline import Source, cited_sources

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
)


@dataclass(frozen=True)
class TraceEvent:
    node: str  # one of NODES
    duration_ms: float
    data: dict[str, Any] = field(default_factory=dict)  # JSON-serializable; keys per node above


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

    def node_ms(self, node: str) -> float:
        """Total time spent in `node` across the run."""
        return sum(e.duration_ms for e in self.trace if e.node == node)
