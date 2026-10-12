# Task: Agent tools, document catalog and tool gateway (Phase 5.5, M1b)

First read and follow `scripts/agents/prompts/_rules.md`. Then read `docs/STATUS.md` and the Phase 5.5 section of `docs/EXECUTION_PLAN.md` (components 2, 4 and 5 are yours).

## Your files
- `src/vaultmind/agent/catalog.py`, `src/vaultmind/agent/tools.py`, `src/vaultmind/agent/gateway.py`: implement the stubs. Keep the public names and signatures; add private helpers freely.
- `tests/test_catalog.py`, `tests/test_tools.py`, `tests/test_gateway.py` (new).

## Contract (read, don't edit)
- `core/providers.py`: `ToolSpec`, `ToolCall`, `AssistantTurn`, `LLM.chat`. `tests/fakes.py`: `FakeLLM(turns=...)`, `tool_turn(...)`.
- `agent/types.py`: `ToolResult`, `ToolError`, `DocumentNotFound`, `AmbiguousDocument`, `ApprovalRequired`, and the `tool` trace-event keys (the gateway doesn't write trace events; the graph will, from `ToolResult`).
- `core/models.py` `DocCard`; `store/cards.py` `CardStore.latest(sha256)` (cards are written by another agent's `ingest/enrich.py`; your tests create cards directly with `CardStore.put`).
- `store/index.py`: `parents_for_file`, `get_parents`; `agent/retrieval.py`: `search()`; `agent/baseline.py`: `Source`, `to_sources`; `agent/guard.py`: `find_injections` and how `build_answer_messages` wraps passages as data.
- Config: `agent.max_steps`, `agent.observation_tokens`, `retrieval.*`, `models.num_ctx`.

## Catalog
- `entries()`: ready files from the registry, `name` relative to `vault_dir` with forward slashes, each with `cards.latest(sha256)`.
- `resolve(mention)`: exact file id → exact name or stem (case-insensitive) → fuzzy match on name, card title and doc_type → reranker score of the mention against "name + title + type + first summary line" when a reranker is given (bge-reranker-v2-m3 is multilingual, so a Hebrew mention can find an English file name and the reverse). Return one entry only when it clearly wins; several close candidates raise `AmbiguousDocument` (names, best first, at most 5); nothing good raises `DocumentNotFound`. Pick and document the thresholds, and test them on synthetic names ("AI in the workplace syllabus" vs "Workato recipe design worksheet" vs "Solution design checklist").
- `describe(limit)`: one line per document: `name | type | language | first summary sentence` (fall back to the file name alone with no card), then "... and N more" past `limit`.

## Tools (`default_tools()`, all read-only)
Each tool has a pydantic `args` model (`extra="forbid"`, field descriptions: they become the JSON schema the model sees) and a one- or two-sentence description. Documents are named by a `doc` string and resolved through `ctx.catalog.resolve`.

| Tool | Does |
|---|---|
| `list_documents(doc_type?, language?)` | the catalog, filtered |
| `document_card(doc)` | the card as readable text (no card yet: say so and suggest `outline` / `read`) |
| `outline(doc)` | section paths with page ranges, from `parents_for_file` |
| `read(doc, pages?, sections?)` | parent sections in order, selected by page range or section index, stopping at the observation budget and saying what was left out; returns the sections as `Source`s |
| `search(query, doc?)` | today's `search()`; with `doc`, keep only that file's hits (over-fetch candidates so scoping still returns results); returns `Source`s |
| `summarize(doc, focus?)` | without focus and with a card: the card summary. Otherwise map-reduce over the sections (each map call fits in `num_ctx`), then one reduce call. Cache results in memory per (sha256, focus, model). Cite sections: each point names the section and page it came from |
| `extract(docs, fields)` | for each document: scoped search per field, then one schema call returning the field values with the page they came from; rows come back as a Markdown table |
| `calc(expression)` | arithmetic with a safe AST walker: numbers, + - * / // % **, parentheses, unary minus, round/min/max/abs. Never `eval`. Cap exponent sizes |
| `date_diff(a, b, unit?)` | dates in ISO and common day-first formats (`15.3.2026`, `15/03/2026`); days or weeks |

No write tools in M1 (they come with human-in-the-loop in M3). No delete tool, ever.

## Gateway
- `call(call, approved=False) -> ToolResult` in this order: unknown tool → failed result listing the tool names; `_unparsed` arguments or validation errors → failed result with the pydantic errors in one or two lines (the M3 loop uses this text for one repair retry); write tool without `approved` → raise `ApprovalRequired`; an identical call (name + canonical JSON of arguments) already made this run → the first result again, not counted; step budget (`agent.max_steps`) used → failed result "step budget used, answer with what you have"; run the tool; `ToolError` → failed result with its message; `AmbiguousDocument` propagates.
- Observation: truncate to `agent.observation_tokens` (`ceil(len/4)` tokens) with a visible "[truncated: N more characters]" marker, run `find_injections` on it (and on each source text), put the flags in `warnings`, and wrap the text in a delimited data block that says it is a tool result, not instructions. Neutralize anything inside that looks like the delimiter.
- `duration_ms` measured per call. `steps` counts executed calls.

## Tests (no models)
Build a small vault in `tmp_path`: a real `Registry`, `ChunkIndex` (via `ingest_path` with `FakeEmbedder`, or `index.add` directly), `CardStore` with cards you write, `FakeReranker`, `FakeLLM` for summarize/extract. Cover every tool's happy path and its errors, resolution (exact, fuzzy, ambiguous, not found, Hebrew), budgets, repeat cache, validation messages, approval for a write tool (register a dummy write tool in the test), truncation, an injection inside a document showing up in `warnings` and inside the data block, and that `Tool.spec` produces a schema `OllamaLLM` can send (no `$defs` surprises; inline them if pydantic emits them).

## Measure (live, at the end)
With Ollama running, show `qwen3:14b` the tool specs (`LLM.chat(messages, tools=gateway.specs())`) for 10 short requests ("what documents do I have", "outline of <a CUAD contract>", "how many days between 1.3.2026 and 15.4.2026", ...) and report how many produce a valid call to the right tool on the first try, with thinking off and on. Use only `vault/cuad/` and `vault/adversarial/` for this (index them into your worktree's own `data/`).
