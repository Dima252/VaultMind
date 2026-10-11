# CLAUDE.md

VaultMind is a fully local, privacy-first agentic RAG desktop app. The product idea (offline, private, corrective RAG over the user's documents) is fixed. Implementation choices may be challenged if there's a measured reason.

- **Start here:** [docs/STATUS.md](docs/STATUS.md) covers what's done, what's next, known limitations and environment gotchas. Update it when you finish a chunk of work.
- Roadmap and current phase: [docs/EXECUTION_PLAN.md](docs/EXECUTION_PLAN.md). Tick its checkboxes when a task lands.
- Eval numbers: `docs/RESULTS.md` (created in Phase 2).
- Dev machine is Windows (console cp1255): keep CLI output ASCII. If `ollama` isn't on PATH, it's at `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`.

## Commands

```bash
uv sync                          # install
uv run pytest                    # unit tests (fakes, no Ollama)
uv run pytest -m ollama          # integration tests (needs Ollama + models)
uv run ruff check . && uv run ruff format .
uv run mypy src tests            # strict
uv run vaultmind doctor          # check Ollama + models
```

Add dependencies with `uv add <pkg>` (dev: `uv add --dev`), never by hand-editing versions. Python 3.12. Don't upgrade to 3.14 until torch/lancedb wheels are reliable.

## Architecture rules

- **No network except localhost.** Never add code that reaches the internet at runtime: no telemetry, no CDN assets, no web-search fallback. Model downloads happen only during explicit setup steps.
- **All model calls go through the protocols** in `src/vaultmind/core/providers.py`. Pipeline code never imports `ollama` or `sentence_transformers` directly.
- **The engine never imports the UI.** `ui/` calls into `ingest/`, `store/`, `agent/`, and nothing depends on `ui/`. There is no HTTP API layer; the NiceGUI app calls the engine in-process.
- **Every pipeline behavior that affects quality is a config switch** in `core/config.py` + `config/default.yaml` (the two must stay in sync; a test enforces it), so the eval runner can ablate it.
- **Grading uses reranker scores, not per-chunk LLM calls.** LLM calls are for analyze, rewrite, generate, and the optional groundedness check.
- **Always pass `num_ctx`** to Ollama; its default silently truncates context.
- Storage: LanceDB holds chunks/vectors/FTS only; SQLite holds files, jobs, chats, traces.
- Answers cite only graded passages. When nothing passes, say "not found in your documents" rather than generating anyway.

## Parallel agents

Larger phases are split across agents working in separate git worktrees (`scripts/agents/`: `setup.ps1` → `start.ps1` → `merge.ps1`). Task prompts live in `scripts/agents/prompts/`. Shared contracts (`core/models.py`, `core/providers.py`, `core/config.py`, stub signatures) are fixed **before** agents start. Agents never edit shared files; they write contract change requests in their commit message instead.

## Conventions

- `src/` layout, type hints everywhere (mypy strict), ruff for lint and format.
- Tests use the fakes in `tests/fakes.py`. Anything needing a real model gets `@pytest.mark.ollama`.
- Match the style of the surrounding code: short module docstrings, small functions, no speculative abstractions beyond the current phase.
- Never commit user documents, indexes, or `config/local.yaml` (all gitignored).
