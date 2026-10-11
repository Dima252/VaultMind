# Rules for every parallel agent

You are one of four agents working **in parallel**, each in its own git worktree and branch. The branches get merged into `main` afterwards, so staying inside your lane is what keeps the merge clean.

1. Read `CLAUDE.md` first and follow it.
2. **Edit only the files listed under "Your files"** in your task. You may create the test files listed there.
3. **Do not edit shared files:** `pyproject.toml`, `uv.lock`, `CLAUDE.md`, `README.md`, `docs/`, `scripts/`, `config/`, `src/vaultmind/core/models.py`, `core/config.py`, `core/providers.py`, `tests/fakes.py`, `tests/__init__.py`, `cli.py`.
4. **Keep the public signatures in your stub files.** You may add private helpers, new exception classes in your own files, and optional keyword-only parameters. If a shared contract genuinely has to change, don't change it. Describe the change under "Contract change requests" in your final summary.
5. **Don't add dependencies.** Everything Phase 1 needs is already installed (`ollama`, `pymupdf4llm`/`pymupdf`, `lancedb`/`pyarrow`, `pydantic`, `pyyaml`). If you're blocked without one, say so in your summary.
6. **No network at runtime.** Code must not reach anything except the configured Ollama host. Tests must pass with no network and no Ollama, except tests marked `@pytest.mark.ollama`.
7. Use the fakes in `tests/fakes.py` where you need a model. Import them with `from tests.fakes import ...` or the relative form `from .fakes import ...`.
8. **Definition of done.** All of these pass from the repo root:
   ```
   uv run ruff format .
   uv run ruff check .
   uv run mypy src tests
   uv run pytest
   ```
9. **Commit on your current branch.** Stage only your files (`git add <paths>`, never `git add -A`). Use a message like `feat(store): LanceDB chunk index and SQLite registry`. Don't push, merge, or switch branches.
10. **Finish with a short summary:** what you built, design decisions worth knowing, known limitations, and contract change requests (or "none"). Put the same summary in the **commit message body**, so the integrator can read it from `git log`.
