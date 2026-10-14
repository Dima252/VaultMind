# VaultMind: Status and Handoff

_Last updated: 2026-10-14 (Phase 5.5 M2 ran live for the first time: routing and the lookup regression pass, whole-document answer quality doesn't yet. Nothing from this session is committed. Start at "Where we stopped")._

This is the starting point for anyone (human or agent) continuing the work. Read it after `CLAUDE.md`. The roadmap with checkboxes is in [EXECUTION_PLAN.md](EXECUTION_PLAN.md), every measured number is in [RESULTS.md](RESULTS.md), and the day-by-day detail is in the git log.

## Where things stand

| Phase | State |
|---|---|
| 0. Project setup | ✅ Done. Eval corpus: 25 CUAD contracts, 10 adversarial documents, 12 of the user's own PDFs |
| 1. Ingestion + baseline RAG | ✅ Done and verified against real models |
| 2. Evaluation harness | ✅ Done. Baseline in RESULTS.md, reproducible. `eval generate` deferred |
| 3. Retrieval quality | ✅ Done. Hybrid + rerank + child chunks are the default |
| 4. Agent loop + safety | ✅ Done. LangGraph loop, injection guard, contradiction and groundedness checks, traces. Only the parts that earned their keep are on by default |
| 5. Desktop app | ✅ Built (NiceGUI, `vaultmind ui`), runs on the real vault. Opens as a browser tab by default. Hand walkthrough partly done (chat page only) |
| 5.5 Document agent | 🟡 **In progress.** M0 (contracts) and M1 (document cards, tools + gateway, task eval) done. **M2 (planner + recipes) runs live: routing and the lookup gate pass; whole-document answer quality and the recorded `--repeat 3` runs are pending (2026-10-14, uncommitted)** |
| 6–7 | ⬜ Not started |

## Where we stopped (2026-10-14)

**M2 ran live on the Windows machine for the first time.** The planner, routes and lookup regression work; whole-document answer quality (summarize, compare, extract) doesn't yet. Nothing from this session is committed: the working tree holds all the changes below, and 906 unit tests, ruff and strict mypy pass on it.

**What M2 does** (`agent/planner.py`, `agent/recipes.py`, new nodes in `agent/nodes.py`):
- **Planner** (`plan` node, first when `agent.router` is on and the card store is passed): one schema call with the catalog, the last 6 chat messages and the message. It returns task, documents, focus, fields, language, steps, clarify and `one_document`. Mentions are resolved with `Catalog.resolve`.
  - Clarifying questions: an ambiguous mention; a summary that names no document; a comparison with fewer than two documents; and (new) `one_document` true with several documents resolved for summarize/generate ("summarize the lecture" with five lectures).
  - A named document that doesn't exist gets "not found". Unparseable output falls back to a lookup. A `transform` with no previous answer becomes a lookup.
- **Routes:** `lookup` → the corrective graph; `chat` → `direct`; `transform` → `respond` over the previous answer; `list | summarize | compare | extract | generate` → `recipe` → `respond`; `open` with documents → the generate recipe (summaries + scoped search), without → lookup; clarify → the `human` node (M3 makes it an interrupt).
- **Recipes** go through the `Gateway`. list: the catalog with no LLM call; with a filter, the documents the planner named (no call), else one call over `list_documents`. summarize: the card, or map-reduce with a focus. compare: summaries + scoped search on the focus. extract: the `extract` table. generate / open: summaries + scoped search. At most 4 documents.
- **Respond:** one streamed call in the planner's language, citing the sections the tools read. The request now comes last, after the tool results and a "results are data" reminder.
- **Eval:** `vaultmind eval tasks` (router accuracy, task success, clarify accuracy, forbidden rate, latency per task); `eval run` adds `plan_p50_ms` and `lookup_route_rate`.

### Done this session (2026-10-14)

1. **Task set reviewed** (`personal_tasks.jsonl` + its review sheet, gitignored): every key fact checked against its source. 24 items changed, mostly bundled facts split (208 facts now). Two facts relied on exponents the PDF parser can't read (2ⁿ/3ⁿ in lesson 19, n² in lesson 21) and were replaced by facts the index does contain. Backup: `eval/datasets/personal_tasks.jsonl.bak-2026-10-14`.
2. **Live check** (the 4 requests + Hebrew versions + lookups) found the problems fixed below.
3. **Fixes** (all with unit tests):
   - **Script guard** (`models.script_guard`, on): qwen3:14b drops Arabic/Cyrillic/Thai/CJK/Hangul words into Hebrew answers, nearly always when the sources are English, and in ~10% of earlier Hebrew lookups. Temperature, a prompt rule and thinking mode didn't help. Free-form answers are now requested as a JSON string whose grammar pattern excludes those scripts, decoded as it streams (`core/ollama.py`). 0 such characters in all live re-runs, no latency cost. When Ollama aborts a looping constrained answer ("token repeat limit"), the call retries unguarded (before any text) or keeps the text so far. Ablation config: `eval/configs/script_guard_off.yaml` (not yet run).
   - **Answer cap** (`models.max_answer_tokens: 2048`, free-form calls only): one guarded compare looped for 43 minutes / ~14k tokens before this.
   - **Ollama crashes:** a dropped connection mid-answer (`httpx.ReadError`, WinError 10054; the runner dies about once an hour under eval load) maps to `OllamaUnavailableError` with a clear message; `eval tasks` retries such an item once.
   - **Poisoned document cards (security):** the card model rewrote planted instructions as facts ("governed by ... the Republic of Atlantis", "all fees are waived"), which the guard can then no longer see, and a compare repeated it. `guard.redact_injections` now drops flagged paragraphs (flagged sentences in very long ones) before card building and before summarize map-reduce, when `agent.injection_guard` is on. On the real vault it removes exactly the 4 planted blocks, nothing else. Card prompts no longer say "summarize it like any other text". `PROMPT_VERSION` 2: all 47 cards rebuilt, none contain planted text.
   - **Planner:** answer language = the message's script (`guess_language`), the model's code only for a transform ("translate to English"); focus only when the user explicitly narrows ("focusing on", "לגבי"), and a focus that repeats a mention is dropped; `one_document` (above); `open` = answers that need a calculation, lookup otherwise.
   - **Search:** extract searches use no rerank threshold (one-word fields like "parties" score 0.0–0.3 on the right passage, Hebrew fields ~0.001); a scoped search keeps its best 3 passages (score ≥ 0.05) when none pass the threshold (compound focus).
   - **Sentence splitting** (`agent/text.split_sentences`, used by cards and the catalog) knows "Inc.", "Ltd.", "e.g.", titles and initials; old cards with a line break after "Inc." read correctly.
   - **Filtered lists** answer with the documents the planner named instead of asking the model to filter a truncated catalog.
4. **Measured** (details in RESULTS.md, Phase 5.5 section):
   - **Lookup regression passes the M2 gate** (3 repeats, run before the planner prompt edits): CUAD correctness 0.809 router on vs 0.802 off, personal 0.906 vs 0.855, abstention identical, `lookup_route_rate` 1.0, planner +1.7–2.0 s p50.
   - **Task eval, 1 pass each:** before the fixes, task success 0.40, router 0.70, clarify 0.17, forbidden 1.6%. After: success **0.38**, router **0.84**, clarify **0.50**, forbidden **0**. Per task after: list 2/5, summarize 4/10, compare 1/8, extract 2/8, transform 7/8, open 4/15, clarify 3/6, injection 1/3.
   - Planner alone after the last prompt edits (`scripts/dev/plan_probe.py`): clarify 4/6, open routed 8/15, lookups stay lookups on CUAD 60/60 and personal 25/27.

### Next session, start here

**Plan changed (2026-10-14, end of session):** follow the course correction in EXECUTION_PLAN.md, Phase 5.5 ("read, don't route"). Steps 0–7 each have a measured gate. The diagnosis is that whole-document answers are built from the 8-line card and 2,000-token tool results, while most documents fit in context whole (median ~3.8K tokens). The list below is the fallback if step 1's gate fails; items 1–3 are folded into steps 1–3.

1. **Fix the list regression** (5/5 → 2/5): for "which documents are about X" the planner no longer names the matches (probably since the "one document → leave documents empty" rule), so the recipe falls back to `list_documents`, whose output (~8,200 chars) is cut by `agent.observation_tokens` before the personal files. Planned fix, not applied: in `PLAN_SYSTEM`'s `documents` line add "For list with a description, every listed document that matches it". Check with `uv run python scripts/dev/plan_probe.py list`. Also consider a `list_documents` budget that fits the whole catalog.
2. **Whole-document answers miss specific facts** (the main gap: summarize 4/10, compare 1/8, extract 2/8). The judge wants exact values ("20%", "Θ(m log n)", "USD 14,200"); card summaries are too general to carry them. Read the per-item answers in `eval/results/20261014-180050/` and try having summarize/compare read sections (a scoped search on the request, or map-reduce) instead of only the card; check what extract returns now that it reads passages. Watch latency (targets in EXECUTION_PLAN.md).
3. **Clarify misses in Hebrew** ("סכם את המטלה", "סכם את השיעור באלגוריתמים"): `one_document` comes back false.
4. **Open tasks** (4/15) need calculation: that's M3's tool loop with `calc`. Don't tune the M2 recipe further.
5. Then the recorded runs: `uv run vaultmind eval tasks --repeat 3` → RESULTS.md, and the lookup regression again (the planner prompt changed since it passed): `uv run vaultmind eval run -d cuad -d personal -c config/default.yaml -c eval/configs/router_off.yaml --repeat 3`. Optionally the script-guard ablation (`-c eval/configs/script_guard_off.yaml`).
6. **You:** the app walkthrough of a transform follow-up ("make it shorter") in `vaultmind ui`.
7. Commit (nothing from 2026-10-14 is committed), tick M2 in EXECUTION_PLAN.md, move to M3.

Ollama must be running (`%LOCALAPPDATA%\Programs\Ollama\ollama.exe` if not on PATH). A task-eval pass takes ~10–15 min with the GPU to itself; `vaultmind enrich` for all 47 cards ~20–40 min. Don't run live checks while an eval runs: they share the GPU and skew latency.

## What works today

```powershell
uv sync --extra rerank                  # torch cu128 + sentence-transformers (plain `uv sync` on the Mac)
uv run vaultmind pull-reranker          # one-time download (the only Hugging Face network step)
uv run vaultmind doctor                 # Ollama reachable, both models pulled, reranker cached
uv run vaultmind ingest [PATH]          # index vault/ (or PATH); incremental
uv run vaultmind enrich                 # write missing document cards (the app does this by itself)
uv run vaultmind ask "question" [--trace]   # planner, then lookup graph or recipe: streamed answer, [n] citations, stored trace
uv run vaultmind ui [--native]          # the app: a browser tab on 127.0.0.1 (or its own window)
uv run vaultmind trace [RUN_ID] [--last N]  # stored runs and their traces
uv run vaultmind config                 # effective config
uv run vaultmind eval import-cuad       # CUAD subset -> vault/cuad/ + eval/datasets/cuad.jsonl
uv run vaultmind eval install-adversarial  # poisoned + contradicting docs -> vault/adversarial/
uv run vaultmind eval run -d cuad -d adversarial -d personal [-c a.yaml -c b.yaml] [--no-judge] [--limit N] [--repeat N]
uv run vaultmind eval tasks [-d personal_tasks] [-c a.yaml] [--task summarize] [--no-judge] [--limit N] [--repeat N]
uv run vaultmind eval sweep eval/results/<run>/<config>__cuad.jsonl -d cuad   # threshold replay
```

Verified on the dev machine (Windows 11, RTX 5070 Ti 16 GB, Ollama with `qwen3:14b` + `qwen3-embedding:0.6b`), up to M1:

- Unit tests run with fakes, no GPU: 885 (2026-10-13, all passing on macOS; last full run on Windows was at M1, 837 tests). Live tests (8) run with `uv run pytest -m ollama` and load the reranker too; they're excluded from the default run and from CI.
- Ingesting 3 Markdown docs → 27 chunks in about 2 s. Re-running skips unchanged files.
- `ask` answers a lookup in about 1.5 s (median) with warm models and `think: false`, plus ~7 s per invocation to load the reranker. It cites the file and section. Out-of-corpus questions get "I couldn't find this in your documents." with the closest matches. Passages with instruction-like text get a `[!]` warning.
- The app: engine load, first token and re-indexing numbers are in RESULTS.md (Phase 5). Served on 127.0.0.1 only, refuses other websites (Host/Origin check), no external assets.
- ruff, ruff format, and mypy `--strict` are clean. CI (`.github/workflows/ci.yml`) runs the same checks plus unit tests.

**Eval data** (all gitignored, on the Windows machine only): CUAD's `data.zip` is in `eval/corpora/cuad/` (18 MB, copied from the `../cuad` repo); the full release with the 510 original PDFs is `eval/corpora/cuad/CUAD_v1.zip` (md5 verified), for the optional txt-vs-PDF comparison. `eval import-cuad` gives the same 25-contract subset for the same seed; ingesting it takes about 20 s. `eval/datasets/personal.jsonl` (27 questions) and `personal_tasks.jsonl` (63 task items, 31 in Hebrew) quote the user's documents. `eval/datasets/tasks_example.jsonl` is a synthetic format example. CUAD is CC BY 4.0: credit it (Hendrycks et al., NeurIPS 2021, The Atticus Project) in RESULTS.md and the README.

## Module map

| Module | What it does | Notes |
|---|---|---|
| `core/config.py` + `config/default.yaml` | Typed config (pydantic, `extra="forbid"`). Resolution: `--config` → `$VAULTMIND_CONFIG` → `config/local.yaml` → `config/default.yaml` | A test asserts the YAML equals the built-in defaults; change both together |
| `core/models.py` | Shared dataclasses: `Block`, `ParsedDocument`, `ParentSection`, `ChildChunk`, `SearchHit`, `FileRecord`, `DocCard`, `file_id_for()` | The contracts between modules |
| `core/providers.py` | `LLM`, `Embedder`, `Reranker` protocols; `LLM.chat(messages, tools=, think=) -> AssistantTurn` with `ToolSpec` / `ToolCall` and a `tool` message role | Pipeline code never imports `ollama` directly. `FakeLLM(turns=...)` and `tool_turn()` in `tests/fakes.py` play back scripted tool calls |
| `core/rerank.py` | `CrossEncoderReranker` (sentence-transformers, bge-reranker-v2-m3, fp16 on CUDA, sigmoid scores in [0, 1]); `download()` / `is_downloaded()` | Loads with `local_files_only` + `HF_HUB_OFFLINE=1`; downloading is only `vaultmind pull-reranker`. Needs the `rerank` extra. Load ~7 s, ~1.1 GB VRAM, ~60 ms per 20 pairs |
| `core/ollama.py` | `OllamaLLM`, `OllamaEmbedder`, `OllamaUnavailableError` | Always sends `num_ctx` + `keep_alive`. Schema calls use `format=`, `temperature 0`, `think=False`. Strips `<think>` text, even when split across stream deltas. Batches embeddings (32) |
| `ingest/parsers.py` | PDF (pymupdf4llm, `page_chunks=True`, no OCR), Markdown, text → `ParsedDocument`. Right-to-left (Hebrew) pages are rebuilt from glyph positions with a bidi pass; RTL tables keep their column order | Heading path carries across PDF pages. Scanned, encrypted and non-UTF-8 files and garbled hidden OCR layers raise `ParseError` |
| `ingest/chunker.py` | Small-to-big chunking: parents ≈1200 tokens, children ≈256 tokens, 15% overlap. Tokens = `ceil(len/4)` | Children are exact slices of parent text. `embed_text` = `Title > Section\n\n` + text, without repeating the title |
| `ingest/pipeline.py` | `ingest_path()`: sha256 skip, parse → chunk → embed → `delete_file` + `add`, removes deleted files; `forget_file` | Parse failures are recorded and **retried** next run. Other errors (Ollama down) mark the file failed and stop the run |
| `ingest/queue.py`, `ingest/watcher.py` | `IngestQueue` (one worker thread, in memory; `JobEvent`s queued/indexing/ready/failed/removed; survives Ollama errors), `VaultWatcher` (watchdog, debounced, ignores temp/hidden files) | `queued` is emitted on the caller's thread, the rest on the worker |
| `ingest/enrich.py` | `build_card` (section summaries in batches that fit half of `num_ctx`, then one card call; language detected from the script, Hebrew if ≥30% of letters, and named in the prompt), `Enricher` (background worker, `paused()` while a question runs) | Started by `Engine.start_ingest` when `ingest.enrich`; `vaultmind enrich` from the CLI |
| `store/index.py` | LanceDB tables `children` (vectors), `parents`, `meta`; `vector_search`, `fts_search` (BM25 over `embed_text`), `refresh_fts`, `parents_for_file` | Cosine similarity score. Refuses a different embedder (`EmbedderMismatchError`). Upserts by id. No telemetry (checked). Native FTS, simple tokenizer + built-in English stemmer (no downloads); rows added since the last refresh are still searched; ingest refreshes it, and the first `fts_search` builds it on an old index |
| `store/registry.py` | SQLite `files` table, WAL, thread-safe, `PRAGMA user_version` migrations; `migrate()` is public | Migration 2: `runs` + `trace_events`; 3: `chats` + `messages`; 4: `doc_cards` |
| `store/traces.py` | `TraceStore`: `runs` + `trace_events` in `vaultmind.sqlite` | `ask` and the app save every loop run; `vaultmind trace` reads them |
| `store/chats.py` | `ChatStore`: chats + messages; assistant messages keep sources, abstention, contradictions, unsupported claims and the run id | Auto-titles a chat from its first question |
| `store/cards.py` | `CardStore`: document cards, keyed by (sha256, model, prompt version); `latest`, `prune` | |
| `agent/types.py` | `AgentResult`, `TraceEvent` (and each node's trace data keys, including `plan`, `recipe`, `reason`, `tool`, `human`, `respond`), `Contradiction`, `UnsupportedClaim`, `TaskKind`, `Plan`, `ToolResult`, the tool exceptions (`AmbiguousDocument`, `DocumentNotFound`, ...) | Shared contract for the graph, checks, traces, eval and CLI |
| `agent/retrieval.py` | `search()`: vector or hybrid (vector + BM25, `rrf_fuse`, k=60) candidates → cross-encoder `rerank` → drop below `rerank_threshold` → top_k | Reranks on `embed_text` (title > section + text). Can return nothing (→ `NOT_FOUND` without an LLM call) |
| `agent/baseline.py` | Linear RAG (`agent.loop: false`): `search()` → `to_sources()` (children, or parents with `small_to_big`) → numbered prompt → answer | `Source` (with `chunk_id`, `warnings`), `NOT_FOUND`, `is_abstention()`, `cited_sources()` |
| `agent/graph.py` + `agent/nodes.py` | LangGraph: [plan] → lookup: analyze (standalone query from history; optional variants) → retrieve (each query, fused with RRF) → grade (rerank against the standalone question, threshold) → [rewrite] → [contradictions] → generate (streamed), not_found or direct → [groundedness]; or recipe / human / direct → respond. `stream_agent` yields tokens, then one `AgentResult` | Analyze is skipped when there's no history and `query_variants: 0` (the default). LangSmith tracing is forced off |
| `agent/planner.py` | `plan()`: one schema call (catalog + history + message) → `Plan`; resolves mentions, asks when ambiguous, reports missing documents, falls back to lookup | M2. Prompt (`PLAN_SYSTEM`) tuned live 2026-10-14 (`scripts/dev/plan_probe.py` runs only the planner) |
| `agent/recipes.py` | `run()`: fixed tool sequences per task through the gateway → `Gathered`; `document_list()`; `respond_messages()` for the cited answer | M2. Prompts (`INSTRUCTIONS`, `RESPOND_SYSTEM`) live since 2026-10-14; answers still too general for specific facts |
| `agent/catalog.py` | `Catalog`: ready files + cards; `resolve(mention)` (exact id/name/title → fuzzy words → the multilingual reranker; close calls raise `AmbiguousDocument`, none `DocumentNotFound`); `describe()` for the planner | |
| `agent/tools.py`, `agent/gateway.py` | 9 read-only tools (list_documents, document_card, outline, read, search with an optional document scope, summarize with cited sections, extract, calc, date_diff); `Gateway`: argument validation, step budget (`agent.max_steps`), repeat cache, observation budget (`agent.observation_tokens`), injection guard, approval for writes | Used by the recipes; the open loop is M3. No write or delete tools yet. Tool specs are flattened (see problems) |
| `agent/guard.py` | `find_injections()`: 18 regexes, English and Hebrew, after NFKC normalization and zero-width stripping; `INJECTION_PATTERNS`. `build_answer_messages()`: delimited `<source>` blocks, neutralized template tokens, warning headers, the contradiction instruction | With the guard off, the prompt is exactly Phase 3's |
| `agent/checks.py` | `detect_contradictions()`, `check_groundedness()`: one schema call each, one retry, then `CheckError` | No call when a check can't apply (one file only; an abstention) |
| `agent/trace_text.py` | One-line trace summaries shared by the CLI and the app | |
| `eval/dataset.py` | `EvalItem` (with a `history` of `Turn`s for follow-ups) + JSONL load/save with line-numbered errors | Validator: unanswerable ⇒ no source; answerable ⇒ `source_file` required. `history` is left out of the JSONL when empty |
| `eval/tasks.py` | `TaskItem`, `load_tasks`, `save_tasks`: the task eval format | |
| `eval/metrics.py` | `recall_at_k`, `mrr`, `abstention_accuracy`, `percentile`, `source_matches`, `evidence_recall_at_k`, `evidence_mrr`, `injection_resistance`, `correctness_score`, plus router and task metrics | Docstrings state exact formulas (cite them in the README) |
| `eval/judge.py` | Correctness judge (correct / partial / incorrect + reason), `judge_key_facts` (fixed-length list, per-fact found + the answer's own words as evidence), `judge_clarify`. One schema call each, one retry, `JudgeError` | Prompts in module constants. Agreement numbers in RESULTS.md |
| `eval/runner.py` | `run_config` (linear or agent, by `agent.loop`), `compute_metrics` (plus `rewrite_rate`, `rewrite_recovered`, `analyze_p50_ms`, `plan_p50_ms`, `lookup_route_rate`), `check_compatible`, `save_report`, `load_results`, `format_table`, `format_repeat_table` (mean and min-max over `--repeat N`) | For agent runs, generation = the generate and direct nodes and retrieval = everything else. Only graded passages count as retrieved |
| `eval/task_runner.py` | `run_tasks()` over a task set: route, key-facts / clarify judging, forbidden strings, latency per task; `save_task_report` | M2; `vaultmind eval tasks` |
| `eval/sweep.py` | Replays `rerank_threshold` over a run saved with threshold 0 (no model calls) | `vaultmind eval sweep RESULTS -d cuad` |
| `eval/configs/*.yaml` | Ablation configs (partial YAML over the defaults) | Phase 3: baseline, hybrid, rerank, hybrid_rerank, hybrid_rerank_children (all pinned to `loop: false`). Phase 4: linear, loop_variants_rewrite, loop_no_variants, loop_no_rewrite, loop_no_guard, loop_contradictions, loop_groundedness |
| `eval/adversarial.py` + `eval/fixtures/adversarial/` | 4 poisoned contracts (override, fake system block, exfiltration, subtle) + 3 contradiction pairs; `install_adversarial()` | Fictional names, `.invalid` domains. Payload codes like `ZORBLAX-4417` must never appear in answers |
| `eval/cuad.py` | CUAD importer: seeded contract subset -> `vault/cuad/*.txt`, questions that name their contract, spans as `evidence`, absent clauses as unanswerable (only when `RELATED` categories are empty too) | Owns `vault/cuad/`: stale `.txt` files there are removed on re-import |
| `app/engine.py` | `Engine`: models, index and stores loaded once; `ask(chat_id, question)` streams tokens then the stored `ChatMessage` (run + exchange saved; nothing on error); `start_ingest` (queue + scan + watcher + `Enricher`); `warm_up` | One question at a time (lock); the enricher pauses while it runs |
| `app/export.py`, `app/highlight.py`, `app/privacy.py`, `app/settings.py` | Markdown/PDF export (Hebrew RTL, system font with Hebrew glyphs); PDF page render with the passage highlighted in memory (and a highlighted copy for the system viewer); connection check of the app's process tree; `local.yaml` overrides + restart/reindex classification | Highlighting found 147/151 real chunks (117/121 Hebrew) |
| `ui/` | NiceGUI: `app.py` (shell, startup, WebView2 lockdown), `state.py` (`AppState`, `off_loop`), `chat.py`, `trace_view.py`, `citation_view.py`, `library.py`, `settings_page.py`, `badge.py`, `local_only.py` (Host/Origin check) | Only `ui/` imports NiceGUI and nothing imports `ui/` (`tests/test_layering.py`) |
| `cli.py` | `config`, `doctor`, `pull-reranker`, `ingest`, `enrich`, `ui [--native/--browser]`, `ask [--trace]`, `trace`, `eval import-cuad`, `eval install-adversarial`, `eval run`, `eval tasks`, `eval sweep` | Forces `errors="replace"` on stdout (Windows cp1255 console) |

Storage lives in `paths.data_dir` (default `./data`): `index/` (LanceDB) and `vaultmind.sqlite`. Both are gitignored.

## Decisions already made (don't re-litigate without measurements)

- **No API layer.** The UI (NiceGUI) calls the engine in-process. The engine never imports `ui/`.
- **Browser tab by default, not the native window** (`ui.native: false`, 2026-10-12). In native mode the window's WebView2 (Microsoft Edge runtime) opens HTTPS connections to Microsoft servers by itself; switching off its background services and a dead local proxy removed the network-service traffic, but its main process still connects through Windows' own networking. In browser mode VaultMind's process tree is one Python process with no outside connections (verified; the badge is green). The trade-off: the browser can see the page, so **use a separate browser profile with no extensions** for VaultMind, and don't use "Translate this page", AI sidebars or enhanced spell check on its tab. `--native` stays available.
- **LanceDB** over Chroma (built-in full-text search for hybrid retrieval, no telemetry). **SQLite** for app state.
- **Reranker scores grade chunks**, not per-chunk LLM calls (Phase 3/4).
- **The LLM reads the graded child chunks, not their parent sections** (`small_to_big: false`, decided 2026-10-12 from 3 runs each: about +8 points correctness, faster). Re-measure on long-form documents.
- **`rerank_threshold` 0.3** is a cheap off-topic filter, not an abstention signal on CUAD. The LLM does the abstaining.
- **Agent defaults (2026-10-12, from the Phase 4 measurements):** `loop: true`, `query_variants: 0`, `rewrite: false`, `injection_guard: true`, `detect_contradictions: true` (runs only when sources span 2+ files), `groundedness_check: false` (+0.5 s, little value). Variants and rewrites cost 1–3 s and didn't improve answers on any dataset. Re-enable them only with a measured reason (M4 grades against several phrasings, which could give them a job).
- **Decided for 5.5 (user, 2026-10-12):**
  1. **Citations:** summaries, comparisons, extractions and generated material cite the sections and pages a tool read. Lookups keep the graded-passage rule and "not found"; a document that doesn't exist gets "not found". In `CLAUDE.md`.
  2. **General knowledge:** strict "only my documents" by default; `agent.world_knowledge` (off) allows clearly labeled answers from the local model.
  3. **Document cards at ingest:** on (`ingest.enrich: true`), in the background, once per file content.
- **Document agent shape (2026-10-12):** a ReAct loop with human-in-the-loop behind fast paths: lookups keep the corrective graph, common tasks run as fixed recipes, only open multi-step tasks enter the tool loop. Still LangGraph: the graph grows a planner and a tool loop, and the lookup graph is reused as one path. Plan and rationale in EXECUTION_PLAN.md, Phase 5.5.
- **Reranker deps are an optional extra** (`rerank`), so CI doesn't download torch.
- **No `tiktoken`**: it downloads its tokenizer at runtime, which breaks the offline rule.
- **`think: false` by default.** Qwen3 thinking adds latency. It's a config switch, so the eval can measure its effect.
- **Python 3.12** (3.14 wheels for torch/lancedb aren't reliable).
- **The name is VaultMind** (the original brief called it H.I.M.). **MIT license.** The repo is **public**.
- **No MCP**, as client or server (decided 2026-10-11). Third-party tool servers would handle document content outside the privacy guarantee. Output features such as PDF export are built in directly.
- **Security as a feature** (added 2026-10-11): prompt-injection hardening with a measured resistance rate, a live "localhost only" privacy badge, a secret/PII scanner at ingest, highlighted citations, and an optional contradiction check. Each is a task in its phase in the plan.

## What's left, in order

1. **Phase 5.5: document agent** (milestones in EXECUTION_PLAN.md): ~~decisions~~, ~~M0~~, ~~M1~~ → **M2 planner + recipes: quality fixes + recorded runs pending** (see "Next session" above) → M3 ReAct loop + human-in-the-loop + verify → M4 retrieval upgrades → M5 UI.
2. **Finish the hand walkthrough** of the app: drop a PDF in, ask, open a citation, open the trace, export, change a setting. Only the chat page has been tried by hand so far.
3. **Phase 6: hardening** (tasks in EXECUTION_PLAN.md): network-blocked test, telemetry audit, launch token (other local programs can use the app today), atomic re-index, Ollama-down recovery, resumable ingestion, edge-case files, the secret/PII scanner. OCR needs a decision (new dependency + model download; offline Hebrew handwriting OCR is weak).
4. **Phase 7: packaging and showcase.** One-command install, an optional Windows build, a demo GIF, the results write-up in the README, CV bullets from RESULTS.md.
5. **Eval quality items, any time:**
   - Harder follow-up (pronoun-only) and multi-hop questions. Today's 4 follow-ups are answered correctly even without history.
   - A stronger injection set, with payloads in the same chunk as the clause asked about. The current set is at 8/8 even without the guard.
   - `vaultmind eval generate` (LLM-drafted questions for human review). Low priority.
   - Optional: CUAD as PDFs (`CUAD_v1.zip`) to measure the txt-vs-PDF parsing gap.
   - (The key-facts judge and `eval run --repeat N` landed in M1; multi-phrasing grading and the refusal second pass are M4.)

## Known limitations

- **Document agent (M2, measured 2026-10-14):**
  - The planner adds one LLM call to every message, lookups included: +1.7–2.0 s p50 on lookups. `qwen3:4b` for the planner is an option to measure.
  - Card-based summaries and compares are too general for specific facts (numbers, percentages, complexities): the main M2 quality gap.
  - qwen3:14b's Hebrew is weak when it writes from English sources: besides the foreign-script words (now blocked by `models.script_guard`) it coins wrong Hebrew words and drops Latin fragments. A model with stronger Hebrew is worth measuring.
  - Ollama 0.40.3's model runner crashes about once an hour under eval load (connection reset mid-answer); it restarts by itself.
  - Lookups ignore the documents the planner resolved: no search scoped to a named file yet (a candidate for M4).
  - Nothing reads `agent.world_knowledge` or `agent.think_open` yet (M3).
  - A clarifying question is just the answer. The user's reply is a new message the planner reads with the history; the `interrupt()` flow is M3.
  - `open` with documents is a fixed recipe, not a reasoning loop (M3). A recipe reads at most 4 documents.
  - The planner sees only the first `agent.catalog_files` (60) catalog lines; the rest are only counted (a named document is still matched against every file), and nothing searches the cards above that yet.
  - The respond call doesn't run the groundedness check (M3 "verify").
  - The list answer (no filter) shows each card's first summary line in the document's own language.
- **Document agent (M1 parts):**
  - Scoped search over-fetches (10× candidates, max 200) and filters by file; a `file_id` prefilter in `ChunkIndex.vector_search` / `fts_search` would be exact (open contract change request).
  - The summarize cache is in memory only.
  - Latin-script documents count as English for cards.
  - The handwritten Hebrew PDF in the real vault is rejected at ingest (no usable text layer), so it has no card.
- **Answer quality (end of Phase 4; numbers in RESULTS.md):**
  - The model refuses when a clause is worded differently from the question (a "most favored nation" clause written as "on terms not less favorable"), even with the passage in front of it. Most remaining CUAD errors are of this kind.
  - With only short chunks in context, it sometimes stretches a related clause into an answer to a question the document can't answer.
  - The reranker scores passages against the standalone question only, so English questions about Hebrew text and two-part questions can fail grading (in the personal set, the right passage scored 0.05–0.12). M4 targets this.
  - The direct (no-retrieval) path relies on the analyze call's `needs_retrieval`. The direct reply refuses factual questions, so a misclassified question gets "not found" instead of a search.
- **App (Phase 5):**
  - The index has no locking: a question asked in the instant a file is being re-indexed (between dropping its old chunks and adding the new ones) can miss that file. Not seen in a live stress test (12 questions during re-indexing).
  - Library "Remove" deletes the file from the vault folder permanently (no recycle bin).
  - During a folder scan the header says "Indexing 1 file": only the current file is known.
  - Settings doesn't expose `agent.loop`, `agent.smalltalk`, `chunking.*` or any Phase 5.5 switch (`agent.router`, `ingest.enrich`, ...).
  - Highlighting misses some math-heavy slides (4 of 151 real chunks) and marks whole lines for right-to-left matches.
  - The privacy badge covers the app's process tree only; Ollama is a separate process.
  - Other programs on the same machine can use the app (no launch token yet, Phase 6). Other websites are refused.
- **Parsing:** code fences are tracked per PDF page; setext (`===`/`---`) headings aren't recognized; sentence splitting knows only a short abbreviation list (`agent/text.py`); PDF headings are only detected when their font is larger. UTF-8 only, no OCR (Phase 6).
- **Hebrew parsing:** the bidi pass is a heuristic.
  - Formula-only lines on Hebrew pages are read left to right.
  - Side-by-side text boxes on one baseline merge into one row.
  - Some math subscripts become `?`.
  - "Hebrew, number, English" sequences can come out in visual order.
- **Index:** exact vector scan with no ANN index (fine at this scale). `add` writes parents and children in two steps, not one transaction.
- **FTS tokenizer:** LanceDB tokenizers that load language dictionaries (jieba, lindera) may **download** them. We use the simple tokenizer with the built-in English stemmer; keep it that way.
- **CLI:** `ask` loads the cross-encoder per invocation (~7 s); the app keeps it loaded. `trace RUN_ID` needs the full id, and stored runs are never pruned.
- **Eval:** ranks aren't deduplicated before the cut at k. Every JSONL line must include `source_file` and `source_page` keys (they can be `null`). Two multi-hop items in `personal.jsonl` span two files but list only one `source_file`, so file-level recall slightly understates them.
- **Abstention detection** is a prefix match on `NOT_FOUND`. If the model rephrases it, it counts as an answer: 2 of 60 CUAD answers in the baseline run.
- A file whose H1 equals its file name gets a label like "CLAUDE.md, CLAUDE.md".
- Embedding batch size (32) isn't in the config yet.
- `vault/README.md` gets indexed like any document, so it shows up as a (harmless) distractor. Consider skipping it in ingest.
- CUAD contract names come from EDGAR filer names (`Aspirityholdingsllc Outsourcing Agreement (2012)`), which often differ from the party names inside the contract. Questions find the right contract mostly through the title prefix on every chunk. That's realistic (users refer to files by name), but keep it in mind when reading results.

## Environment gotchas

### Windows dev machine (RTX 5070 Ti; live checks and evals)

- **Console code page is cp1255.** Keep CLI output ASCII (`[ OK ]`/`[FAIL]`, not ✓/✗). `cli.py` sets `errors="replace"` as a safety net.
- **`ollama` may not be on PATH** in a new shell. The executable is `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`.
- **Line endings:** `core.autocrlf=true`, so the "LF will be replaced by CRLF" warnings are harmless.
- **RTX 50 series (Blackwell):** torch comes from the `cu128` index (`[tool.uv.sources]` in `pyproject.toml`), only with `uv sync --extra rerank`. CI installs without the extra; mypy treats torch/sentence-transformers as untyped there.
- **`uv add` re-syncs without the `rerank` extra and uninstalls torch.** Run `uv sync --extra rerank` after adding dependencies (it's in every agent prompt).
- **VRAM:** `qwen3:14b` (11 GB at `num_ctx` 16384) + the embedder (2.4 GB) + the reranker (1.1 GB) fit in 16 GB, all on GPU.
- Live tests (`-m ollama`) are excluded from default `pytest` runs and from CI.

### MacBook (M1, no NVIDIA GPU; code and unit tests only)

- No Ollama, no vault, and no personal eval files live there. `qwen3:14b` needs ~10 GB of RAM, so the live checks and evals belong on the Windows machine.
- Plain `uv sync` is enough for unit tests. The CUDA index in `pyproject.toml` applies only on Windows and Linux, so `--extra rerank` here would install the regular PyPI torch (CPU/MPS). It hasn't been tried yet.
- **Python skips `.pth` files with the macOS "hidden" flag**, so `import vaultmind` fails inside `uv run`. Fix: `chflags nohidden .venv/lib/python3.12/site-packages/*.pth`. The flag came back several times in one session, so rerun it whenever `import vaultmind` fails. If that gets tedious, put the venv outside `~/Documents` with `UV_PROJECT_ENVIRONMENT`.
- macOS FSEvents replays recent creations (the vault root, files written just before the watch starts) when a watch starts. The watcher ignores them (`ingest/watcher.py`, with a test), so the watcher tests pass here too.

## Problems we hit and how they were handled

| Problem | What we did |
|---|---|
| torch on an RTX 50 (Blackwell) needs CUDA 12.8 builds, and CI shouldn't download 3 GB of torch | torch comes from the `cu128` index via `[tool.uv.sources]`. The reranker deps are an optional `rerank` extra, and mypy treats them as untyped in CI |
| The reranker would download itself from Hugging Face on first use, which breaks "offline" | It loads with `local_files_only` + `HF_HUB_OFFLINE=1`. Downloading happens only in the explicit `vaultmind pull-reranker` |
| langgraph pulls in langchain-core, which can upload traces to LangSmith | `LANGSMITH_TRACING` and `LANGCHAIN_TRACING_V2` are forced to `false` in `agent/graph.py`, and a test checks it |
| Generation got ~0.4 s slower with the reranker loaded | Measured: no VRAM contention in isolation, all three models fit on the 16 GB GPU, and context size was equal. Accepted and recorded |
| The rerank threshold was meant to drive abstention, but CUAD's unanswerable questions still score high (they name a real contract) | Swept it offline (`eval sweep`). 0.3 stays as a free filter for off-topic questions (they score <0.02); the LLM does the abstaining |
| Small-to-big (sending parent sections) measured worse than sending the graded chunks | Off by default after 3 runs each: about +8 points correctness without it |
| The agent answered "What is the capital of Peru?" with "Lima" from world knowledge (the no-retrieval path) | Analyze now treats every information request as needing retrieval, and the direct reply refuses factual questions with the not-found answer |
| Changing defaults silently changed what old ablation configs measured | Phase 2/3 configs pin `agent.loop: false` (and `small_to_big: true` where needed) |
| Hebrew PDFs came out reversed (words, brackets, dates like "159..2026") | The parser rebuilds right-to-left pages from glyph positions with a bidi pass (tests use synthetic text only) |
| A 69 MB handwritten summary produced 85K characters of junk from a garbled hidden OCR layer | Detected and rejected with "no usable text layer" in 1.4 s |
| The user's documents and questions must never reach the public repo | `eval/datasets/personal*` is gitignored, tests use synthetic Hebrew, and agent commits were reviewed for quoted content |
| Worktree agents start from the session's first commit, not current `main` | Every agent prompt starts with `git merge --ff-only main` |
| Eval artifact: a not-found answer's closest matches were counted as "retrieved", which made the loop look perfect on the personal set | The runner now scores only passages that passed grading |
| NiceGUI binds `0.0.0.0` outside native mode, which would expose the app (and your documents) to the LAN | `ui.run(host="127.0.0.1")` always; checked live that the LAN address can't connect |
| `uv add` re-syncs without the `rerank` extra and uninstalls torch | Run `uv sync --extra rerank` after adding dependencies (in every agent prompt) |
| NTFS updates access times about hourly, so the startup scan's reads came back as "modified" events and the watcher resubmitted every file (seen live: 72 events on an unchanged vault) | The watcher records existing files' (mtime, size) on start and only submits real changes |
| The native window's WebView2 connects to Microsoft servers by itself | Background networking off + dead proxy; the main process still connects. Decided: browser mode by default (see decisions) |
| **Other websites could read and drive the app.** 127.0.0.1 keeps the network out, but not a malicious page in a local browser: the server answered a forged Host header (DNS rebinding) with the full library page, and NiceGUI's socket.io allows any origin. Affects native and browser mode alike | `ui/local_only.py`: ASGI middleware that returns 403 unless Host is our address and Origin (when sent) is our page. Verified live: forged Host and foreign-origin socket.io/websocket requests get 403, our own page works |
| First hand test: "HI" got "not found" plus three unrelated closest matches. Analyze is skipped on a chat's first message, so nothing classified it as a greeting | `agent.smalltalk` (on): when nothing passes grading on an unclassified message, one short direct reply; factual questions still get not-found with the closest matches. Live: greetings (English, Hebrew) answered, "capital of Peru" and Bellman-Ford stay not-found |
| First hand test: the chat column was ~230 px wide, the answer and chips cut off, the user's own message invisible | NiceGUI's sub-pages container shrink-wraps its page; scroll-area content could outgrow the window; a Tailwind color that doesn't exist. Fixed and checked with headless Edge screenshots at 600 and 1280 px |
| Merged UI tests waited for "any chip" to mean "the answer arrived", but the header's privacy badge is a chip too | Tests use the citation chips' marker |
| First hand test of the agent (2026-10-12): "summarize X", "what documents do I have" and "compare X and Y" all failed, because every message took the lookup path | Phase 5.5, before hardening. The four requests and why they failed are in EXECUTION_PLAN.md (Phase 5.5); they are M2's live check |
| Left to itself, qwen3 wrote Hebrew summaries of English contracts, or English summaries of Hebrew lectures | Document cards detect the language from the script and name it in the prompt |
| The ollama client silently drops `anyOf` / `$ref` / `default` from tool schemas | Tool specs are flattened (`flat_schema`), and a test checks them |
| `calc` on a 1500-term sum hit the recursion limit (found in review) | An input-length cap |

## How the work was split (for reuse)

Larger phases were built by parallel agents, each in its own git worktree, after the shared contracts and stub signatures were fixed on `main`; an integration pass on `main` then wired everything together. All merges were conflict-free.

| Phase | Agents (prompts in `scripts/agents/prompts/done/`) | On `main` before / after |
|---|---|---|
| 1 | providers, ingest, store, eval (+ integrate) | contracts and stubs; integration pass |
| 2 | runner, judge, adversarial | shared metrics and CLI |
| 3 | none | built on `main` |
| 4 | graph, safety, traces, personal | contracts commit `5051fb2` |
| 5 | wave 1: queue, chats, citations; wave 2: chat_ui, library_ui | contracts + `Engine`; the UI shell between waves; final integration |
| 5.5 M1 | cards, tools, taskeval | M0 contracts; Enricher in `Engine`, `vaultmind enrich` |
| 5.5 M2 | none | built on `main` by one session |

- Task prompts: `scripts/agents/prompts/` (`_rules.md` holds the shared rules; finished prompts move to `prompts/done/`).
- The agents were run with Claude Code's built-in subagents (`isolation: worktree`). The PowerShell scripts in `scripts/agents/` (`setup.ps1` → `start.ps1` → `merge.ps1`) do the same thing by hand. They pass a syntax check but **have not been run end to end yet**.
- **Gotcha:** Claude Code's worktree agents branched from the commit the session started on, not current `main`, so each had to `git merge --ff-only main` to see the contracts. That step is rule 0 in `_rules.md`; keep it in the launch prompt.
- To reuse for a later phase: write new stubs and contracts first, add one prompt per agent, and set `$Agents` in `scripts/agents/_common.ps1`.
