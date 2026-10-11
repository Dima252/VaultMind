# Rules for every parallel agent

You are one of several agents working **in parallel**, each in its own git worktree and branch. The branches get merged into `main` afterwards, so staying inside your lane is what keeps the merge clean.

0. **Sync first.** Your worktree may have been created from an older commit. Before anything else, run `git merge --ff-only main` and check that `src/vaultmind/agent/types.py` exists. If the merge fails, stop and report it.
1. Read `CLAUDE.md` first and follow it.
2. **Edit only the files listed under "Your files"** in your task. You may create the test files listed there.
3. **Do not edit shared files** (unless your task lists one under "Your files"): `pyproject.toml`, `uv.lock`, `.gitignore`, `CLAUDE.md`, `README.md`, `docs/`, `scripts/`, `config/`, everything in `src/vaultmind/core/`, `agent/types.py`, `agent/baseline.py`, `agent/retrieval.py`, `store/index.py`, `store/registry.py`, `tests/fakes.py`, `tests/__init__.py`, `cli.py`, `eval/dataset.py`, `eval/metrics.py`, `eval/runner.py`, and any file another agent owns.
4. **Keep the public signatures in your stub files.** You may add private helpers, new exception classes in your own files, and optional keyword-only parameters. If a shared contract genuinely has to change, don't change it. Describe the change under "Contract change requests" in your final summary.
5. **Don't add dependencies.** Everything this phase needs is already installed: `ollama`, `pymupdf4llm`/`pymupdf`, `lancedb`/`pyarrow`, `pydantic`, `pyyaml`, `typer`, `langgraph`. The reranker's torch + sentence-transformers come with `uv sync --extra rerank`. If you're blocked without a dependency, say so in your summary.
6. **No network at runtime.** Code must not reach anything except the configured Ollama host. Tests must pass with no network and no Ollama, except tests marked `@pytest.mark.ollama`.
7. **Gitignored folders are not in your worktree:** `vault/`, `data/`, `eval/results/`, `eval/corpora/`, `config/local.yaml`. The real ones are in the main checkout at `C:/Users/dimat/Documents/GitHub/VaultMind`. Read them there by absolute path only when your task says so, and never write to the main checkout's `data/`.
8. Use the fakes in `tests/fakes.py` where you need a model. Import them with `from .fakes import ...`.
9. **Definition of done.** All of these pass from the repo root of your worktree:
   ```
   uv run ruff format .
   uv run ruff check .
   uv run mypy src tests
   uv run pytest
   ```
10. **Commit on your current branch.** Stage only your files (`git add <paths>`, never `git add -A`). Use a message like `feat(agent): corrective LangGraph loop`. Don't push, merge, or switch branches. The repo is **public**: never put text from the user's own documents in commits, tests or code.
11. **Finish with a short summary:** what you built, design decisions worth knowing, known limitations, and contract change requests (or "none"). Put the same summary in the **commit message body**, so the integrator can read it from `git log`.
