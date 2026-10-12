# Task: Library page, settings page, privacy badge, queue indicator (Phase 5, wave 2)

First read and follow `scripts/agents/prompts/_rules.md`. Then read `docs/STATUS.md` and the Phase 5 section of `docs/EXECUTION_PLAN.md` (including "Rules for the UI").

## Your files
- `src/vaultmind/ui/library.py`, `src/vaultmind/ui/settings_page.py`, `src/vaultmind/ui/badge.py`: implement the stubs, keep the public functions.
- `tests/test_ui_library.py` (new).

## Contract (read, don't edit)
- `ui/app.py` (shell: the header calls `library.queue_indicator()` and `badge.privacy_badge()` on every page; `library.page` runs only once the engine is ready; `settings_page.page` runs even when the engine failed) and `ui/state.py` (`state().engine`, `state().cfg`, `state().config_source`, `state().jobs()` = latest `JobEvent` per path since startup, filled from the ingest worker thread). `tests/test_ui_shell.py` shows how to drive pages with `nicegui.testing.user_simulation(root=...)` and `@pytest.mark.anyio`.
- `app/engine.py`: `engine.registry.all()` (`FileRecord`: path, status, error, indexed_at), `engine.queue` (`IngestQueue`: `submit`, `forget`, `scan`, `pending`), `engine.index.count()`.
- `ingest/parsers.py`: `SUPPORTED_EXTENSIONS`. `app/privacy.py`: `check()` → `PrivacyStatus`. `app/settings.py`: `overrides`, `save_local`, `needs_restart`, `needs_reindex`. `core/config.py`: `Config`, `LOCAL_CONFIG`.

## Library page (`page()`)
- Summary line: N files, N ready, N failed, N chunks in the index, and the vault folder path.
- Drag-and-drop upload (`ui.upload`, multiple, auto upload) restricted to `SUPPORTED_EXTENSIONS`: save each file into `cfg.paths.vault_dir` (keep the name; if it exists with different content, add ` (2)`, ` (3)`...), then `engine.queue.submit(path)` (the watcher would catch it too after the debounce, but submitting right away feels faster; double submits collapse). Reject other types with a notification. Large files: don't hold more than one in memory at a time.
- File table: name, folder (relative to the vault), status, error (for failed), indexed time (local). Status merges the registry with live `state().jobs()` (the newer wins: a `queued`/`indexing` job event shows even when the registry still says `ready`). A filter box and a status filter. Refresh with a `ui.timer` (about 1 s) that only rebuilds when something changed.
- Actions per file: "Re-index" (`queue.submit`; to force re-indexing an unchanged file, set its registry status to `queued` via `registry.set_status(file_id, "queued")` first, because unchanged `ready` files are skipped), "Open" (the OS default app), "Remove" (confirm dialog that says the file will be **deleted from the vault folder**; delete it, then `queue.forget`). Only offer Remove for files inside the vault folder.
- "Re-scan vault" button (`queue.scan()`).

## Queue indicator (`queue_indicator()`)
- In the header: a spinner plus "Indexing N files" while `engine.queue.pending() > 0`, hidden otherwise, and nothing while the engine isn't ready. Clicking it navigates to `/library`. Poll with a `ui.timer`.

## Privacy badge (`privacy_badge()`)
- A chip in the header: green "localhost only" when `check().local_only`, red "N remote connections" otherwise, tooltip listing `remote` and the time of the last check. Run `check()` off the event loop (`run.io_bound`) every `cfg.ui.privacy_check_s` seconds. Clicking it opens a dialog that explains what is checked (this process and its children, e.g. the window's browser processes; Ollama runs as a separate process and isn't covered) and lists the remote addresses.

## Settings page (`page()`)
- Sections with inputs bound to a working copy of the config:
  - Documents: vault folder (text input; validate that it exists or can be created), watch on/off, debounce.
  - Models: LLM, embedder, reranker names, `num_ctx`, thinking on/off. Show the names Ollama has pulled if you can list them cheaply via `ollama.Client(host=cfg.ollama.host).list()` off the loop (it's localhost); fall back to free text when Ollama is down.
  - Retrieval: hybrid, rerank, candidates, top_k, rerank_threshold, small_to_big.
  - Agent: query_variants, rewrite, max_rewrites, injection_guard, detect_contradictions, groundedness_check.
  - Window: native, port, privacy check interval.
  Use the config field comments in `config/default.yaml` as short help texts.
- "Save": validate (`Config.model_validate`; show field errors), `save_local`. If `state().config_source` is not `LOCAL_CONFIG` (a `--config` or `$VAULTMIND_CONFIG` file is active), say that the saved file won't be used until they launch without it. Then:
  - `needs_reindex` → warn that the index must be rebuilt (suggest deleting `data/index` and restarting; don't do it automatically).
  - `needs_restart` → "Restart VaultMind to apply: <keys>".
  - otherwise apply immediately to the running app: `state().cfg = new` and `engine.cfg = new` (the next question uses it), and say "Applied".
- "Reset to defaults" (with confirmation) fills the defaults into the form; nothing is saved until Save.

## Tests (no models)
- Use a real `Registry`, `ChunkIndex`, and a fake or real `IngestQueue` with `FakeEmbedder`, in `tmp_path`, wired into an `Engine` (see the chat UI prompt pattern: fakes, `retrieval.rerank: false`) or a minimal stand-in object with the attributes the page uses. Set `ui_state.init_state(cfg)` with `phase = "ready"`.
- With `user_simulation`: the library lists files with statuses; a live `indexing` job event overrides `ready`; upload saves into the vault and submits (use the user simulation's upload support or call the handler directly); Remove asks for confirmation and deletes only after it; the queue indicator appears with pending jobs; the badge shows "localhost only" with `check` monkeypatched, and the red state with a fake remote. Settings: save writes `local.yaml` under `tmp_path` (monkeypatch `LOCAL_CONFIG` or pass the path through your own module constant) with only the changed keys; a models change reports a restart; a retrieval change applies to `engine.cfg`; invalid input shows an error and writes nothing.
