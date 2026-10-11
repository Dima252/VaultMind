# Task: Integrate Phase 1 (run on `main` after merging the four agent branches)

Read `CLAUDE.md` and `docs/EXECUTION_PLAN.md` first. Then review the four merged modules (`core/ollama.py`, `ingest/`, `store/`, `eval/`) and their commit messages for any "contract change requests". Apply the ones that make sense, deliberately, and update every caller.

## Build
- `src/vaultmind/ingest/pipeline.py`: `ingest_path(path, cfg, *, embedder, index, registry)` walks files with supported extensions.
  - For each file: sha256 → skip if the registry has the same hash and status `ready`. Otherwise mark `indexing`, parse, chunk, embed child `embed_text` in batches, `index.delete_file` then `add`, and mark `ready` with the embedder name and time. On `ParseError`, mark `failed` with the reason and continue.
  - Files in the registry that no longer exist on disk get removed from the index and the registry.
  - Return a summary (added / updated / skipped / failed / removed).
- `src/vaultmind/agent/baseline.py`: linear baseline RAG. Embed the question → `vector_search(k=cfg.retrieval.top_k)` → fetch parent sections (deduplicate by parent) → prompt with numbered sources → answer citing `[n]`. Return the answer text plus the list of sources (file path, page, section). Instruct the model to say it can't find the answer when the sources don't contain it.
- CLI in `cli.py`: `vaultmind ingest [PATH]` (default `cfg.paths.vault_dir`) prints the summary; `vaultmind ask "question"` streams the answer, then lists the sources. Storage lives under `cfg.paths.data_dir` (`index/` for LanceDB, `vaultmind.sqlite`).
- Tests with fakes: pipeline skip/update/remove/failed behavior, baseline citation mapping. Add one `@pytest.mark.ollama` end-to-end test over a tiny generated corpus.

## Finish
- All checks pass (`ruff format`, `ruff check`, `mypy src tests`, `pytest`).
- Tick the completed Phase 1 boxes in `docs/EXECUTION_PLAN.md`.
- Commit on `main`.
