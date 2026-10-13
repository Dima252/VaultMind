# CLAUDE.md

VaultMind is a fully local, privacy-first agentic RAG desktop app. The product idea (offline, private, corrective RAG over the user's documents) is fixed. Implementation choices may be challenged if there's a measured reason.

- **Start here:** [docs/STATUS.md](docs/STATUS.md) covers what's done, what's next, known limitations and environment gotchas. Update it when you finish a chunk of work.
- Roadmap and current phase: [docs/EXECUTION_PLAN.md](docs/EXECUTION_PLAN.md). Tick its checkboxes when a task lands.
- Eval numbers: [docs/RESULTS.md](docs/RESULTS.md). Record new measurements there.
- Dev machine is Windows (console cp1255): keep CLI output ASCII. If `ollama` isn't on PATH, it's at `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`. A MacBook without Ollama is used for code and unit tests only (gotchas in STATUS.md).

## Commands

```bash
uv sync --extra rerank           # install (the extra = torch cu128 + sentence-transformers; plain `uv sync` is enough on the Mac)
uv run vaultmind pull-reranker   # one-time reranker download (the only HF network step)
uv run pytest                    # unit tests (fakes, no Ollama)
uv run pytest -m ollama          # integration tests (needs Ollama + models + reranker)
uv run ruff check . && uv run ruff format .
uv run mypy src tests            # strict
uv run vaultmind doctor          # check Ollama + models
```

Add dependencies with `uv add <pkg>` (dev: `uv add --dev`), never by hand-editing versions, then run `uv sync --extra rerank` again (`uv add` re-syncs without the extra and uninstalls torch). Python 3.12. Don't upgrade to 3.14 until torch/lancedb wheels are reliable.

## Architecture rules

- **No network except localhost.** Never add code that reaches the internet at runtime: no telemetry, no CDN assets, no web-search fallback. Model downloads happen only during explicit setup steps.
- **All model calls go through the protocols** in `src/vaultmind/core/providers.py`, tool calling included. Pipeline code never imports `ollama` or `sentence_transformers` directly.
- **The engine never imports the UI.** `ui/` calls into `ingest/`, `store/`, `agent/`, and nothing depends on `ui/`. There is no HTTP API layer; the NiceGUI app calls the engine in-process.
- **Every pipeline behavior that affects quality is a config switch** in `core/config.py` + `config/default.yaml` (the two must stay in sync; a test enforces it), so the eval runner can ablate it.
- **Grading uses reranker scores, not per-chunk LLM calls.** LLM calls are for planning/analyze, rewrite, tool-loop steps, summarize/extract/compare, document cards at ingest, generate, and the optional checks.
- **Fast paths before the tool loop.** Lookups keep the corrective-RAG graph, and list/summarize/compare/extract run as fixed recipes. Only open multi-step tasks enter the ReAct tool loop, with bounded steps (Phase 5.5).
- **Agent tools are plain Python functions** (`agent/tools.py`) over the index, registry and document cards: no MCP, no network. Read-only by default; any write needs human approval; the agent never gets a delete tool. Tool observations go through the injection guard.
- **Always pass `num_ctx`** to Ollama; its default silently truncates context.
- Storage: LanceDB holds chunks/vectors/FTS only; SQLite holds files, chats, traces and document cards (and, from Phase 5.5 M3, agent checkpoints). The ingestion job queue is in memory for now (resumable ingestion is Phase 6).
- Lookup answers cite only graded passages. When nothing passes, say "not found in your documents" rather than generating anyway. Whole-document tasks (summarize, compare, extract, generate) cite the sections and pages a tool actually read. A request about a document that doesn't exist still gets "not found" (decided 2026-10-12).
- Answers come only from the user's documents. An optional, clearly labeled world-knowledge mode (`agent.world_knowledge`, off by default) may answer from the local model's own knowledge.

## Parallel agents

Larger phases are split across agents working in separate git worktrees (`scripts/agents/`: `setup.ps1` → `start.ps1` → `merge.ps1`). Task prompts live in `scripts/agents/prompts/` (finished ones in `prompts/done/`); set the agent list in `_common.ps1`. Shared contracts (`core/models.py`, `core/providers.py`, `core/config.py`, stub signatures) are fixed **before** agents start. Agents never edit shared files; they write contract change requests in their commit message instead.

## Conventions

- `src/` layout, type hints everywhere (mypy strict), ruff for lint and format.
- Tests use the fakes in `tests/fakes.py`. Anything needing a real model gets `@pytest.mark.ollama`.
- Match the style of the surrounding code: short module docstrings, small functions, no speculative abstractions beyond the current phase.
- Never commit user documents, indexes, or `config/local.yaml` (all gitignored).
