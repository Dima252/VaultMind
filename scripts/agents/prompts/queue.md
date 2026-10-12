# Task: Background ingestion queue + live folder watching (Phase 5, wave 1)

First read and follow `scripts/agents/prompts/_rules.md`. Then read `docs/STATUS.md` and the Phase 5 section of `docs/EXECUTION_PLAN.md`.

## Your files
- `src/vaultmind/ingest/queue.py`: implement the stub, keep the signatures.
- `src/vaultmind/ingest/watcher.py`: implement the stub, keep the signatures.
- `tests/test_queue.py`, `tests/test_watcher.py` (new).

## Contract (read, don't edit)
- `ingest/pipeline.py`: `ingest_path(path, cfg, embedder=, index=, registry=, on_file=)` indexes a file or a folder (only new/changed files; for a folder it also forgets deleted files under it). `forget_file(path, index=, registry=)` drops one file. `iter_files`, `SUPPORTED_EXTENSIONS` (from `ingest/parsers.py`).
- `app/engine.py`: `Engine.start_ingest` builds `IngestQueue(cfg, embedder=, index=, registry=, on_event=)`, calls `start()`, `scan()`, then `VaultWatcher(cfg.paths.vault_dir, queue, debounce_s=cfg.ingest.debounce_s).start()`; `Engine.close` calls `watcher.stop()` then `queue.stop()`.
- `core/config.py`: `cfg.ingest.watch`, `cfg.ingest.debounce_s`, `cfg.paths.vault_dir`.

## IngestQueue
- One daemon worker thread; jobs are `("index", path)` or `("forget", path)`, processed in order. `submit` of a path that is already waiting (not running) is a no-op; a file submitted while it's being indexed runs again afterwards (it may have changed mid-indexing).
- Events (`on_event`, called on the worker thread; swallow and log exceptions from the callback so it can't kill the worker):
  - `queued` on `submit` of a single file.
  - `indexing` when a file actually starts (use `ingest_path`'s `on_file`; unchanged files are skipped silently).
  - `ready` / `failed` (with the reason) per file from the `IngestSummary`, and `removed` for forgotten files (`summary.removed` and `forget`).
- Errors: `ingest_path` raises on anything but a parse error (e.g. `OllamaUnavailableError`). Catch it, emit `failed` for the file it was working on with a readable message, and keep the worker alive. The registry already marks the file failed, so the next `scan` retries it.
- `scan()`: submit `cfg.paths.vault_dir` (create it if missing), and forget registry records whose path is outside the vault (the user pointed the vault somewhere else).
- `stop()`: let the current file finish, drop the rest, join. Idempotent; `submit` after `stop` is ignored.
- `pending()` and `wait_idle()` count waiting + running jobs (tests and the UI's "N in queue" use them).
- Thread safety: `Registry` is locked internally. The index is shared with the chat (searches run while the worker writes); don't add locks around whole files (that would block questions for seconds). Note anything you observe in your summary.

## VaultWatcher
- `watchdog.observers.Observer`, recursive on `root`.
- Only files with `SUPPORTED_EXTENSIONS` (case-insensitive). Ignore hidden files and folders (name starts with `.`), Office/editor temp files (`~$*`, `*.tmp`, `*.part`, `*.crdownload`, `*.swp`), and anything outside `root`.
- Debounce per path: a created/modified file is submitted once it has had no new events for `debounce_s` (a copy of a large PDF fires many events). Use one timer thread or a `threading.Timer` per path; cancel on new events.
- Deleted → `queue.forget(path)` (cancel a pending submit for it). Moved → forget the source, debounce-submit the destination (if it's supported and inside `root`). A deleted or moved **folder** → forget every registered file under it; the watcher gets only one event for it on Windows. `queue.scan()` would also handle it, but do it directly; `forget` of an unknown file is harmless.
- `stop()` cancels pending timers and stops/joins the observer. Idempotent.

## Tests (fakes only, no Ollama)
- Queue, using real `ChunkIndex` + `Registry` in `tmp_path` and `FakeEmbedder`: index a folder → events and `ready` records; re-submit unchanged → no `indexing` event; modify → re-indexed; `forget`; `scan` forgets out-of-vault records; duplicate submits collapse; an embedder that raises (`OllamaUnavailableError`) → `failed` event, worker still processes the next job; a raising `on_event` doesn't kill the worker; `stop` is idempotent and drops waiting jobs.
- Watcher, with a fake queue that records calls (and `debounce_s` around 0.2): create → one submit after the burst; rapid modifications collapse to one; delete → forget; rename → forget + submit; unsupported/temp/hidden files ignored; folder delete forgets the files under it. Use a polling helper with a timeout (no fixed long sleeps); keep the whole file under ~10 s. File-system events on Windows can be flaky: assert on eventual state, not exact event counts, where needed.
