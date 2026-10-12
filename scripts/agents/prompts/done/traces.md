# Task: Trace store, `ask --trace`, eval runner loop switch (Phase 4)

First read and follow `scripts/agents/prompts/_rules.md`. Then read `docs/STATUS.md` and the Phase 4 section of `docs/EXECUTION_PLAN.md`.

## Your files
- `src/vaultmind/store/traces.py`: implement the stub, keep the signatures.
- `src/vaultmind/eval/runner.py` (you own it for this phase).
- `src/vaultmind/cli.py` (you own it for this phase).
- `tests/test_traces.py`, `tests/test_cli.py` (new), and `tests/test_eval_runner.py`.

## Contract (read, don't edit)
- `agent/types.py`: `AgentResult`, `TraceEvent`, `NODES`, and the per-node trace `data` keys in its docstring.
- `agent/graph.py`: `stream_agent` / `run_agent`. They are stubs in your worktree (another agent is writing them), so **in tests, monkeypatch `vaultmind.eval.runner.run_agent` and `vaultmind.cli.stream_agent`**. Import them as module-level names; in `cli.py`, a module-level import inside the command that tests can patch is fine, as long as it can be patched.
- `store/registry.py`: `migrate(conn)` creates the `runs` and `trace_events` tables (migration 2). `TraceStore` uses the same SQLite file as `Registry` (`paths.data_dir / "vaultmind.sqlite"`).
- `eval/dataset.py`: `EvalItem.history` (list of `Turn`).

## Trace store
`TraceStore(db_path)`: open with WAL and foreign keys on, call `migrate`, and serialize access with a lock like `Registry`. `save(result)` writes one `runs` row (uuid4 hex id, UTC ISO time, question, answer, abstained) and its events in order (`data` as compact JSON, `ensure_ascii=False`) in one transaction, and returns the id. `get` returns None for an unknown id. `recent(limit)` returns newest first, with events. `close()` is idempotent. Test that it works side by side with a `Registry` on the same file.

## Eval runner
- With `cfg.agent.loop` on, `run_config` calls `run_agent(item.question, cfg, ..., history=[{"role": t.role, "content": t.content} for t in item.history])`. With it off, keep today's linear path unchanged (and ignore history there).
- `run_config` gains the reranker it already has. Map `AgentResult.sources` to `RetrievedSource` the same way as today.
- Timing for the loop: `generation_ms = result.node_ms("generate") + result.node_ms("direct")`, and `retrieval_ms` = total wall time minus `generation_ms`. That keeps "total" honest and puts analyze, rewrite and checks on the retrieval side. Document it on `ItemResult`.
- Add `ItemResult` fields with defaults, so old result files still load in `load_results`: `queries: list[str]`, `rewrites: int`, `node_ms: dict[str, float]` (per-node totals).
- Append to `METRIC_NAMES` (keep the existing order):
  - `rewrite_rate`: share of items with `rewrites > 0`.
  - `rewrite_recovered`: among items with `rewrites > 0`, the share that ended answered (not abstained). None when no item was rewritten.
  - `analyze_p50_ms`: p50 of `node_ms["analyze"]` over items that have it, else None.

  None for linear runs.
- Keep `format_table` working with the extra rows.

## CLI
- `ask`: with `cfg.agent.loop` on, use `stream_agent`. Print each `str` as it arrives. On the final `AgentResult`:
  - save it to the `TraceStore` and print the run id on a short line
  - print `Sources:` with the cited sources (or `Closest matches:` on abstention), as today
  - print the warnings: a source's `warnings` (marked `[!]`), `contradictions`, `unsupported` claims

  With loop off, keep today's behavior exactly.
- `ask --trace`: after the answer, print the formatted trace. One line per event: `node`, duration in ms, then a compact summary of its data (e.g. grade: `3/5 passed (threshold 0.30)` plus top scores; rewrite: the new query). Node lines are plain ASCII; question and passage text may be any language (stdout already uses `errors="replace"`).
- New `vaultmind trace [RUN_ID] [--last N]`: show a stored run's question, answer and formatted trace; without an id, list the last N runs (id, time, abstained, question).
- `eval run`: the progress line shows `rewrites=N` for loop runs. Nothing else changes.
- Keep CLI output ASCII apart from user and document text.

## Tests (fakes only)
- TraceStore: round-trip, ordering, `recent`, unknown id, Unicode (Hebrew) text, coexistence with `Registry`.
- Runner: the loop path with a monkeypatched `run_agent` returning scripted `AgentResult`s: history passed through, timing split, new metrics hand-computed, linear path unchanged, and `load_results` reading an old-format file.
- CLI: use `typer.testing.CliRunner`, with `stream_agent` monkeypatched and a temp config (`paths.data_dir` under `tmp_path`, `retrieval.rerank: false` so no reranker loads). Check that tokens print, sources print, the run is saved, `--trace` output, and `trace` / `trace --last`. You may need to create a small index with `FakeEmbedder` and make `OllamaEmbedder.from_config` / `OllamaLLM.from_config` return fakes via monkeypatch.
