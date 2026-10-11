# Task: Storage (LanceDB index + SQLite registry)

First read and follow `scripts/agents/prompts/_rules.md`.

## Your files
- `src/vaultmind/store/index.py` (implement the stub)
- `src/vaultmind/store/registry.py` (implement the stub)
- `tests/test_index.py`, `tests/test_registry.py` (new)

## Goal
Persistent local storage. `ChunkIndex` holds child chunks with their vectors and the parent sections. `Registry` tracks which files are indexed so ingestion can skip unchanged files. Types are in `src/vaultmind/core/models.py`. Check the installed `lancedb` API in `.venv` rather than assuming it.

## ChunkIndex (LanceDB, embedded, directory `db_dir`)
- Tables:
  - `children`: id, parent_id, file_id, text, embed_text, page (nullable int), section_path (list of strings), vector (fixed-size float32 list)
  - `parents`: id, file_id, text, section_path, page_start, page_end
  - a small `meta` table: embedder name, vector dimension
- **Embedder guard**: on open, if `meta` records a different `embedder_name`, raise `EmbedderMismatchError` telling the user to re-index. The dimension is fixed by the first `add`. Later vectors with a different dimension raise `ValueError`.
- `add`: validate `len(children) == len(vectors)`. Adding an id that already exists replaces it rather than duplicating it (use merge-insert/upsert, or delete-then-add).
- `vector_search`: cosine metric. `SearchHit.score` = cosine similarity (higher is better). Return `[]` on an empty index. Rebuild `ChildChunk` exactly as stored, including `section_path` as a tuple and `page=None`.
- `delete_file(file_id)` removes the file's rows from `children` and `parents`. Validate `file_id` (hex/`:` only) before putting it in a filter string.
- `get_parents` returns only the ids found. `count()` = number of children.
- Store `embed_text` as its own column; Phase 3 adds a full-text index on it.
- **Privacy**: check whether lancedb (or its dependencies) sends telemetry or makes any network call for local use. Disable it if so, and note what you found in your summary.

## Registry (SQLite at `db_path`)
- Table `files` mirrors `FileRecord`. Store `path` as a string and `indexed_at` as ISO-8601 UTC.
- Use `PRAGMA user_version` as a schema version, with a small migration hook for future changes. Turn on WAL mode.
- Safe to use from more than one thread (the UI will run ingestion in the background): one connection with `check_same_thread=False` guarded by a lock.
- `set_status` on an unknown id raises `KeyError`. `all()` is ordered by path. `close()` is idempotent.
- Create parent directories as needed.

## Tests
- Use `FakeEmbedder` from `tests/fakes.py` for vectors and `tmp_path` for storage.
- Index: add + search returns the identical vector's chunk first; scores fall in [-1, 1] and are sorted; reopening the directory keeps the data; a different embedder name raises; a dimension mismatch raises; re-adding the same ids doesn't duplicate; `delete_file` removes only that file; `get_parents`; `page=None` and `section_path` round-trip; search on an empty index.
- Registry: upsert/get round-trip including datetime and None fields; status updates; unknown id raises; ordering; persistence across reopen; concurrent `set_status` from several threads.
