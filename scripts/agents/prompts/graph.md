# Task: Corrective agent loop (Phase 4)

First read and follow `scripts/agents/prompts/_rules.md`. Then read `docs/STATUS.md`, the Phase 4 section of `docs/EXECUTION_PLAN.md`, and the Phase 3 section of `docs/RESULTS.md` (it explains what the loop has to fix).

## Your files
- `src/vaultmind/agent/graph.py`: implement `stream_agent`. Keep both public signatures, `run_agent` as is, and the two `os.environ` lines that force LangSmith tracing off.
- `src/vaultmind/agent/nodes.py` (new): node functions and their prompts.
- `tests/test_graph.py` (new).

## Contract (read these, don't edit them)
- `agent/types.py`: `AgentResult`, `TraceEvent` and the **trace `data` keys for each node** (follow the table in its docstring exactly; the trace store, CLI and eval read them).
- `agent/retrieval.py`: `candidates()`, `rrf_fuse()`, `rerank()`. `agent/baseline.py`: `to_sources()`, `Source` (now with `chunk_id`, `warnings`), `NOT_FOUND`, `is_abstention()`.
- `agent/guard.py`: `build_answer_messages()`, `find_injections()`. `agent/checks.py`: `detect_contradictions()`, `check_groundedness()`, `CheckError`. Another agent is implementing these. `guard.py` holds working placeholders, and `checks.py` raises `NotImplementedError`. **Import them into `nodes.py` as module-level names**, so tests can monkeypatch `vaultmind.agent.nodes.<name>`.
- Config: `cfg.agent.*` (`query_variants`, `rewrite`, `max_rewrites`, `injection_guard`, `detect_contradictions`, `groundedness_check`) and `cfg.retrieval.*`. The `loop` switch is read by callers, not by you.

## Behavior
A LangGraph `StateGraph` with a TypedDict state (question, history, queries tried, candidates, graded sources, rewrite count, trace events, and whatever else you need). No checkpointer, no persistence.

1. **analyze**: one `llm.generate(..., schema=...)` call returning `standalone_query` (resolves follow-ups using `history`), up to `query_variants` alternative phrasings, and `needs_retrieval`. Variants must keep names, codes and document titles from the question verbatim (CUAD questions name their contract; dropping the name breaks retrieval). `needs_retrieval` is false only for greetings or questions about the assistant itself; when in doubt, retrieve. **Skip the LLM call** when there's no history and `query_variants == 0`. On unparseable output, fall back to the raw question, record `error` in the trace, and continue.
2. **direct** (when `needs_retrieval` is false): a short streamed reply that consults no documents and states no facts about them.
3. **retrieve**: `candidates()` for each query of this round, fused with `rrf_fuse`.
4. **grade**: with `retrieval.rerank`, `rerank()` the fused candidates **against the standalone question** (not the variants). A passage passes when its score >= `rerank_threshold`; keep the top `top_k` that pass. Without reranking, keep the top `top_k`, all passing, with threshold None in the trace. Turn passing hits into sources with `to_sources()`. With `injection_guard` on, set `Source.warnings` from `find_injections(source.text)` (`dataclasses.replace`). Remember the best failing hits across rounds for the not-found answer.
5. Route: something passed → contradictions (only if enabled and the sources span 2+ files) → generate. Nothing passed → **rewrite** if `rewrite` is on and rounds remain (`max_rewrites`), else **not_found**.
6. **rewrite**: one schema call returning a new `query`, given the question, the queries already tried, and short snippets of what was found. A query equal to one already tried (case/space-insensitive) ends the loop: go to not_found.
7. **generate**: `llm.stream(build_answer_messages(question, sources, guard=cfg.agent.injection_guard, history=history, contradictions=...))`, with the user's original question. **Yield tokens as they arrive**, not after the node finishes. LangGraph's `get_stream_writer()` with `stream_mode="custom"` is one way to do it. `abstained = is_abstention(answer)`.
8. **not_found**: answer `NOT_FOUND` (yield it as text), `abstained=True`, sources = up to 3 closest matches (best-scoring failed hits, as sources).
9. **groundedness** (if enabled and the answer isn't an abstention): `check_groundedness` → `AgentResult.unsupported`. A `CheckError` is recorded in the trace and doesn't fail the run. Same handling for contradictions.

Every node appends one `TraceEvent` with its duration (`time.perf_counter`). The final `AgentResult` has `queries` (standalone first, then variants and rewrites, in search order), `rewrites`, and the full trace. Pipeline code never imports `ollama` or `sentence_transformers`.

## Tests (fakes only, no Ollama)
Build a real `ChunkIndex` + `Registry` in `tmp_path` like `tests/test_retrieval.py`. Use `FakeLLM` (queued responses; a dict is returned as JSON for schema calls; `stream` splits on spaces), `FakeEmbedder` and `FakeReranker` (word overlap with the query, in [0, 1]). Cover:
- the happy path, with the trace node order and the data keys from `types.py`
- a rewrite that recovers, and rewrites exhausted → not_found with closest matches
- `rewrite: false`, and a repeated rewrite query
- analyze fallback on bad JSON, and the analyze call skipped when it isn't needed
- history reaching both analyze and the answer messages
- the direct path
- rerank off
- injection warnings on sources (monkeypatch `find_injections`)
- contradictions and groundedness on/off, including a `CheckError` (monkeypatch the check functions)
- streaming: at least one str is yielded before the `AgentResult`, and the `AgentResult` comes last
- `run_agent` matches the streamed result
