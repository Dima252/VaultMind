# VaultMind: Status and Handoff

_Last updated: 2026-10-12 (Phase 5 built)._

This is the starting point for anyone (human or agent) continuing the work. Read it after `CLAUDE.md`. The full roadmap with checkboxes is in [EXECUTION_PLAN.md](EXECUTION_PLAN.md).

## Where things stand

| Phase | State |
|---|---|
| 0. Project setup | ✅ Done. Eval corpus: 25 CUAD contracts + 10 adversarial documents. Your own mixed-type documents still optional |
| 1. Ingestion + baseline RAG | ✅ Done and verified against real models |
| 2. Evaluation harness | ✅ Done. Baseline in [RESULTS.md](RESULTS.md), reproducible. `eval generate` deferred |
| 3. Retrieval quality | ✅ Done. Hybrid + rerank + child chunks are the new default; ablations in [RESULTS.md](RESULTS.md) |
| 4. Agent loop + safety | ✅ Done. LangGraph loop, injection guard, contradiction and groundedness checks, traces. Measured in [RESULTS.md](RESULTS.md); only the parts that earned their keep are on by default |
| 5. Desktop app | ✅ Built (NiceGUI, `vaultmind ui`). One open decision: the native window's WebView2 phones Microsoft (see below) |
| 6–7 | ⬜ Not started |

**Phase 5 state:** `uv run vaultmind ui` (native window) or `vaultmind ui --browser`. Built in two waves of parallel agents (plan in EXECUTION_PLAN.md, "Parallel build"). Verified with the real models on the real vault: the engine loads in ~15 s plus ~10 s Ollama warm-up (a loading screen covers it), first token ~2.9 s, follow-ups ~1.6 s; the startup scan of an unchanged vault takes ~2.6 s; 12 questions asked while 9 files re-indexed in a loop all answered correctly with no errors. Served on 127.0.0.1 only (not reachable from the LAN), no external assets.

**Open decision (privacy):** in native mode the window's WebView2 (Microsoft Edge runtime) opens HTTPS connections to Microsoft servers by itself, with only our 127.0.0.1 page loaded. VaultMind's own Python process makes none. The privacy badge shows it red, correctly. `ui/app.py` switches off WebView2's background services and routes the rest to a dead local proxy: that removed the network-service connections, but the main browser process still connects (it uses Windows' own networking). Options: make `ui.native: false` (browser mode) the default, keep native and document it, or block it outside the app (Edge policy or a firewall rule).

**Next:** the user walks through the app by hand (drop files in, watch them index, ask, open a citation, inspect the trace, export); then Phase 6.

The 27 personal reference answers were verified against the source pages (2026-10-12); none needed changing, so the personal-set numbers in RESULTS.md stand.

## Honest assessment (2026-10-12, end of Phase 4)

- **Retrieval (Phase 3) moved the numbers, not the agent loop (Phase 4).** Hybrid search + reranking took CUAD correctness from ~0.64 to ~0.79. The loop's two agentic steps, query variants and rewrites, added nothing measurable to answer quality and cost 1–3 s per question, so they are off by default. They stay in the code behind config switches.
- **What Phase 4 added that works:**
  - Chat follow-ups are rewritten into standalone questions.
  - A question with no relevant passage gets an honest "not found" with the closest matches, and no LLM call.
  - Injected instructions in documents are flagged and neutralized: 40/40 attacks resisted in a targeted test, vs 30/40 without the guard.
  - The optional contradiction check found all 3 planted conflicts, with no false alarms.
  - Every answer has a stored trace you can inspect.
- **Quality today:**
  - CUAD (legal contracts): ~0.80 of answers correct, ~0.86 abstention accuracy.
  - Your own documents: ~0.91 correct (reference answers verified). With 27 items, one question is ~4 points.
  - Median answer time is ~1.5 s on an RTX 5070 Ti.
- **Where it still fails:**
  1. The model refuses when a clause is worded differently from the question (a "most favored nation" clause written as "on terms not less favorable"), even with the passage in front of it.
  2. With only short chunks in context, it sometimes stretches a related clause into an answer to a question the document can't answer.
  3. English questions about Hebrew text, and two-part questions, give the right passage a low reranker score, so it fails grading.
- **How much to trust the numbers:**
  - Retrieval metrics are exact.
  - Correctness comes from an LLM judge and varies about ±5 points between identical runs: two configs that ran the same pipeline scored 0.802 and 0.854. Differences under ~5 points are noise.
  - Several Phase 4 decisions rest on small sets: 3 contradiction items, 8 injection items, 4 follow-ups.

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
| The native window's WebView2 connects to Microsoft servers by itself | Background networking off + dead proxy; the main process still connects. Open decision, see the top of this file |
| Merged UI tests waited for "any chip" to mean "the answer arrived", but the header's privacy badge is a chip too | Tests use the citation chips' marker |

## Redesign considerations (not scheduled; each needs a measurement before it lands)

1. **Grade against more than one phrasing.** Today every passage is scored against the single standalone question, which is why rewrites and variants can't help. A better query finds new candidates, but the right passage is still scored against the original wording. Instead:
   - score each passage by its best score over the question and its variants;
   - for a cross-lingual question, add a translation into the document's language;
   - split two-part questions into sub-questions graded separately.

   This targets failure 3 above, and would give the loop a real job.
2. **Fix the generation side, not retrieval.** Most remaining CUAD errors are refusals with the right passage in context. Two options:
   - a cheap second pass only when the model abstains while graded passages exist ("does any passage address this in other words?");
   - a less literal answer prompt.

   Either is the most likely next gain. Measure abstention on unanswerable questions alongside it.
3. **A smaller model for the side calls.** Analyze, rewrite and the checks don't need a 14B model. `qwen3:4b` or `8b` would make variants and the groundedness check affordable (the plan's risk table already suggests this).
4. **Stronger evals.**
   - The injection set is at its ceiling (8/8 even without the guard), so the guard's value currently rests on a targeted test. Add payloads in the same chunk as the clause being asked about.
   - Add harder follow-ups (pronoun-only) and more multi-hop items.
   - Switch the judge to a key-facts checklist, and average 2–3 repeated runs per config (`eval run --repeat N`), so 5-point effects can be decided.
5. **Keep LangGraph for now.** With variants and rewrites off, the graph is nearly linear. It still pays for itself in streaming, per-node traces and the switches the eval needs. Don't re-architect before Phase 5.

**Eval corpus (2026-10-12):** CUAD's `data.zip` is copied to `eval/corpora/cuad/` (gitignored, 18 MB, from the `../cuad` repo). `uv run vaultmind eval import-cuad` writes 25 contracts to `vault/cuad/` and `eval/datasets/cuad.jsonl`; the same seed always gives the same subset. Ingest takes about 20 s. All 12 unanswerable items were checked by hand against the contract text (two bad labels found and fixed via `RELATED` categories in `eval/cuad.py`). The full CUAD release with the 510 original PDFs is at `eval/corpora/cuad/CUAD_v1.zip` (md5 verified) for the optional txt-vs-PDF comparison. CUAD is CC BY 4.0: credit it (Hendrycks et al., NeurIPS 2021, The Atticus Project) in `RESULTS.md` and the README.

## What works today

```powershell
uv sync --extra rerank                  # torch cu128 + sentence-transformers
uv run vaultmind pull-reranker          # one-time download (the only Hugging Face network step)
uv run vaultmind doctor                 # Ollama reachable, both models pulled, reranker cached
uv run vaultmind ingest [PATH]          # index vault/ (or PATH); incremental
uv run vaultmind ask "question" [--trace]   # agent loop: streamed answer, [n] citations, warnings, stored trace
uv run vaultmind trace [RUN_ID] [--last N]  # stored runs and their traces
uv run vaultmind config                 # effective config
uv run vaultmind eval import-cuad       # CUAD subset -> vault/cuad/ + eval/datasets/cuad.jsonl
uv run vaultmind eval install-adversarial  # poisoned + contradicting docs -> vault/adversarial/
uv run vaultmind eval run -d cuad -d adversarial -d personal [-c a.yaml -c b.yaml] [--no-judge] [--limit N]
uv run vaultmind eval sweep eval/results/<run>/<config>__cuad.jsonl -d cuad   # threshold replay
```

Verified on the dev machine (Windows 11, RTX 5070 Ti 16 GB, Ollama with `qwen3:14b` + `qwen3-embedding:0.6b`):

- 628 unit tests pass (fakes, no GPU). 6 live tests pass with `uv run pytest -m ollama` (they load the reranker too).
- Ingesting 3 Markdown docs → 27 chunks in about 2 s. Re-running skips unchanged files.
- `ask` answers in about 1.5 s (median) with warm models and `think: false`, plus ~7 s per invocation to load the reranker. It cites the file and section. For out-of-corpus questions it answers "I couldn't find this in your documents." and lists the closest matches. Passages with instruction-like text get a `[!]` warning.
- ruff, ruff format, and mypy `--strict` are clean. CI (`.github/workflows/ci.yml`) runs the same checks plus unit tests.

## What was built (module map)

| Module | What it does | Notes |
|---|---|---|
| `core/config.py` + `config/default.yaml` | Typed config (pydantic, `extra="forbid"`). Resolution: `--config` → `$VAULTMIND_CONFIG` → `config/local.yaml` → `config/default.yaml` | A test asserts the YAML equals the built-in defaults; change both together |
| `core/models.py` | Shared dataclasses: `Block`, `ParsedDocument`, `ParentSection`, `ChildChunk`, `SearchHit`, `FileRecord`, `file_id_for()` | The contracts between modules |
| `core/providers.py` | `LLM`, `Embedder`, `Reranker` protocols | Pipeline code never imports `ollama` directly |
| `core/rerank.py` | `CrossEncoderReranker` (sentence-transformers, bge-reranker-v2-m3, fp16 on CUDA, sigmoid scores in [0, 1]); `download()` / `is_downloaded()` | Loads with `local_files_only` + `HF_HUB_OFFLINE=1`; downloading is only `vaultmind pull-reranker`. Needs the `rerank` extra. Load ~7 s, ~1.1 GB VRAM, ~60 ms per 20 pairs |
| `core/ollama.py` | `OllamaLLM`, `OllamaEmbedder`, `OllamaUnavailableError` | Always sends `num_ctx` + `keep_alive`. Schema calls use `format=`, `temperature 0`, `think=False`. Strips `<think>` text, even when split across stream deltas. Batches embeddings (32) |
| `ingest/parsers.py` | PDF (pymupdf4llm, `page_chunks=True`, no OCR), Markdown, text → `ParsedDocument`. Right-to-left (Hebrew) pages are rebuilt from glyph positions with a bidi pass; RTL tables keep their column order | Heading path carries across PDF pages. Scanned, encrypted and non-UTF-8 files and garbled hidden OCR layers raise `ParseError` |
| `ingest/chunker.py` | Small-to-big chunking: parents ≈1200 tokens, children ≈256 tokens, 15% overlap. Tokens = `ceil(len/4)` | Children are exact slices of parent text. `embed_text` = `Title > Section\n\n` + text, without repeating the title |
| `ingest/pipeline.py` | `ingest_path()`: sha256 skip, parse → chunk → embed → `delete_file` + `add`, removes deleted files | Parse failures are recorded and **retried** next run. Other errors (Ollama down) mark the file failed and stop the run |
| `store/index.py` | LanceDB tables `children` (vectors), `parents`, `meta`; `vector_search`, `fts_search` (BM25 over `embed_text`), `refresh_fts` | Cosine similarity score. Refuses a different embedder (`EmbedderMismatchError`). Upserts by id. No telemetry (checked). Native FTS, simple tokenizer + built-in English stemmer (no downloads); rows added since the last refresh are still searched; ingest refreshes it, and the first `fts_search` builds it on an old index |
| `store/registry.py` | SQLite `files` table, WAL, thread-safe, `PRAGMA user_version` migrations; `migrate()` is public | Migration 2 adds `runs` + `trace_events` |
| `agent/retrieval.py` | `search()`: vector or hybrid (vector + BM25, `rrf_fuse`, k=60) candidates → cross-encoder `rerank` → drop below `rerank_threshold` → top_k | Reranks on `embed_text` (title > section + text). Can return nothing (→ `NOT_FOUND` without an LLM call) |
| `agent/baseline.py` | Linear RAG (`agent.loop: false`): `search()` → `to_sources()` (children, or parents with `small_to_big`) → numbered prompt → answer | `Source` (with `chunk_id`, `warnings`), `NOT_FOUND`, `is_abstention()`, `cited_sources()` |
| `agent/types.py` | `AgentResult`, `TraceEvent` (and each node's trace data keys), `Contradiction`, `UnsupportedClaim` | Shared contract for the graph, checks, traces, eval and CLI |
| `agent/graph.py` + `agent/nodes.py` | LangGraph loop: analyze (standalone query from history; optional variants) → retrieve (each query, fused with RRF) → grade (rerank against the standalone question, threshold) → [rewrite] → [contradictions] → generate (streamed), not_found or direct → [groundedness]. `stream_agent` yields tokens, then one `AgentResult` | Analyze is skipped when there's no history and `query_variants: 0` (the default). LangSmith tracing is forced off |
| `agent/guard.py` | `find_injections()`: 18 regexes, English and Hebrew, after NFKC normalization and zero-width stripping. `build_answer_messages()`: delimited `<source>` blocks, neutralized template tokens, warning headers, the contradiction instruction | 0 false positives on 1,188 indexed chunks; all 4 planted injections flagged. With the guard off, the prompt is exactly Phase 3's |
| `agent/checks.py` | `detect_contradictions()`, `check_groundedness()`: one schema call each, one retry, then `CheckError` | No call when a check can't apply (one file only; an abstention) |
| `store/traces.py` | `TraceStore`: `runs` + `trace_events` tables in `vaultmind.sqlite` (registry migration 2) | `ask` saves every loop run; `vaultmind trace` reads them |
| `eval/dataset.py` | `EvalItem` (with a `history` of `Turn`s for follow-ups) + JSONL load/save with line-numbered errors | Validator: unanswerable ⇒ no source; answerable ⇒ `source_file` required. `history` is left out of the JSONL when empty |
| `eval/metrics.py` | `recall_at_k`, `mrr`, `abstention_accuracy`, `percentile`, `source_matches`, `evidence_recall_at_k`, `evidence_mrr`, `injection_resistance`, `correctness_score` | Docstrings state exact formulas (cite them in the README) |
| `eval/judge.py` | LLM judge: one schema call, verdict correct / partial / incorrect + reason, one retry, `JudgeError` | Prompt in module constants. Ignores answer content the reference doesn't cover. 11 of 15 hand-checked verdicts agreed |
| `eval/runner.py` | `run_config` (linear or agent, by `agent.loop`), `compute_metrics` (plus `rewrite_rate`, `rewrite_recovered`, `analyze_p50_ms`), `check_compatible`, `save_report`, `load_results`, `format_table` | For agent runs, generation = the generate and direct nodes and retrieval = everything else. Only graded passages count as retrieved |
| `eval/sweep.py` | Replays `rerank_threshold` over a run saved with threshold 0 (no model calls) | `vaultmind eval sweep RESULTS -d cuad` |
| `eval/configs/*.yaml` | Ablation configs (partial YAML over the defaults) | Phase 3: baseline, hybrid, rerank, hybrid_rerank, hybrid_rerank_children (all pinned to `loop: false`). Phase 4: linear, loop_variants_rewrite, loop_no_variants, loop_no_rewrite, loop_no_guard, loop_contradictions, loop_groundedness |
| `eval/adversarial.py` + `eval/fixtures/adversarial/` | 4 poisoned contracts (override, fake system block, exfiltration, subtle) + 3 contradiction pairs; `install_adversarial()` | Fictional names, `.invalid` domains. Payload codes like `ZORBLAX-4417` must never appear in answers |
| `eval/cuad.py` | CUAD importer: seeded contract subset -> `vault/cuad/*.txt`, questions that name their contract, spans as `evidence`, absent clauses as unanswerable | Owns `vault/cuad/`: stale `.txt` files there are removed on re-import |
| `app/engine.py` | `Engine`: models, index and stores loaded once; `ask(chat_id, question)` streams tokens then the stored `ChatMessage` (run + exchange saved; nothing on error); `start_ingest` (queue + scan + watcher); `warm_up` | One question at a time (lock) |
| `ingest/queue.py`, `ingest/watcher.py` | `IngestQueue` (one worker thread, `JobEvent`s queued/indexing/ready/failed/removed, survives Ollama errors), `VaultWatcher` (watchdog, debounced, ignores temp/hidden files) | `queued` is emitted on the caller's thread, the rest on the worker |
| `store/chats.py` | `ChatStore`: chats + messages (migration 3); assistant messages keep sources, abstention, contradictions, unsupported claims and the run id | Auto-titles a chat from its first question |
| `app/export.py`, `app/highlight.py`, `app/privacy.py`, `app/settings.py` | Markdown/PDF export (Hebrew RTL, system font with Hebrew glyphs); PDF page render with the passage highlighted in memory (and a highlighted copy for the system viewer); connection check of the app's process tree; `local.yaml` overrides + restart/reindex classification | Highlighting found 147/151 real chunks (117/121 Hebrew) |
| `ui/` | NiceGUI: `app.py` (shell, startup, WebView2 lockdown), `state.py` (`AppState`, `off_loop`), `chat.py`, `trace_view.py`, `citation_view.py`, `library.py`, `settings_page.py`, `badge.py` | Only `ui/` imports NiceGUI and nothing imports `ui/` (`tests/test_layering.py`) |
| `agent/trace_text.py` | One-line trace summaries shared by the CLI and the app | |
| `cli.py` | `config`, `doctor`, `pull-reranker`, `ingest`, `ui [--browser]`, `ask [--trace]`, `trace`, `eval import-cuad`, `eval install-adversarial`, `eval run`, `eval sweep` | Forces `errors="replace"` on stdout (Windows cp1255 console) |

Storage lives in `paths.data_dir` (default `./data`): `index/` (LanceDB) and `vaultmind.sqlite`. Both are gitignored.

## Decisions already made (don't re-litigate without measurements)

- **No API layer.** The UI (NiceGUI, `native=True`) will call the engine in-process. The engine never imports `ui/`.
- **LanceDB** over Chroma (built-in full-text search for hybrid retrieval, no telemetry). **SQLite** for app state.
- **Reranker scores grade chunks**, not per-chunk LLM calls (Phase 3/4).
- **The LLM reads the graded child chunks, not their parent sections** (`small_to_big: false`, decided 2026-10-12 from 3 runs each: about +8 points correctness, faster). Re-measure on long-form documents once the mixed-type supplement exists.
- **`rerank_threshold` 0.3** is a cheap off-topic filter, not an abstention signal on CUAD (see RESULTS.md). The LLM does the abstaining.
- **Agent defaults (2026-10-12, from the Phase 4 measurements):** `loop: true`, `query_variants: 0`, `rewrite: false`, `injection_guard: true`, `detect_contradictions: true` (runs only when sources span 2+ files), `groundedness_check: false` (+0.5 s, little value). Variants and rewrites cost 1–3 s and didn't improve answers on any dataset. Re-enable them only with a measured reason (see redesign idea 1).
- **Reranker deps are an optional extra** (`rerank`), so CI doesn't download torch.
- **No `tiktoken`**: it downloads its tokenizer at runtime, which breaks the offline rule.
- **`think: false` by default.** Qwen3 thinking adds latency. It's a config switch, so the eval can measure its effect.
- **Python 3.12** (3.14 wheels for torch/lancedb aren't reliable).
- **The name is VaultMind** (the original brief called it H.I.M.). **MIT license.** The repo is **public**.
- **No MCP**, as client or server (decided 2026-10-11). Third-party tool servers would handle document content outside the privacy guarantee. Output features such as PDF export are built in directly.
- **Security as a feature** (added 2026-10-11): prompt-injection hardening with a measured resistance rate, a live "localhost only" privacy badge, a secret/PII scanner at ingest, highlighted citations, and an optional contradiction check. Each is a task in its phase in the plan.

## What's left

### Open items from earlier phases
- [x] Verify the 27 reference answers in `eval/datasets/personal.jsonl`. Done 2026-10-12, no changes needed. Two multi-hop items span two files but list only one `source_file`, so file-level recall slightly understates them.
- [ ] Harder follow-up (pronoun-only) and multi-hop questions. Today's 4 follow-ups are answered correctly even without history.
- [ ] A stronger injection set, with payloads in the same chunk as the clause asked about. The current set is at 8/8 even without the guard.
- [ ] A judge built on a key-facts checklist, plus `eval run --repeat N`, so ~5-point effects can be decided.
- [ ] `vaultmind eval generate` (LLM-drafted questions for human review). Low priority.
- [ ] Optional: CUAD as PDFs (`CUAD_v1.zip`) to measure the txt-vs-PDF parsing gap.

### Phase 5: Desktop app (next, largest)
`vaultmind ui`, a NiceGUI native window with:
- streamed chat with citation chips, and a trace panel
- a library with per-file status, and live folder watching
- settings
- highlighted citations (PDF page + passage)
- export to Markdown/PDF
- the "localhost only" privacy badge

Load the models once at startup.

### Phase 6: Hardening
- A network-blocked test: ingest + ask with all sockets blocked except 127.0.0.1.
- A telemetry audit. langsmith is installed; tracing is forced off.
- Edge cases: encrypted or huge files, files changing during indexing.
- Recovery when Ollama goes down, and resumable ingestion.
- The secret/PII scanner at ingest, reusing `guard.INJECTION_PATTERNS` for injection flags.
- OCR for scanned and handwritten PDFs.

### Phase 7: Packaging + showcase
One-command install, an optional Windows build, a demo GIF, the final results table and write-up in the README, and CV bullets from RESULTS.md.

## Known limitations (from the agents' reports and testing)

- **App (Phase 5):**
  - The index has no locking: a question asked in the instant a file is being re-indexed (between dropping its old chunks and adding the new ones) can miss that file. Not seen in a live stress test (12 questions during re-indexing).
  - Library "Remove" deletes the file from the vault folder permanently (no recycle bin).
  - During a folder scan the header says "Indexing 1 file": only the current file is known.
  - Settings doesn't expose `agent.loop` or `chunking.*`.
  - Highlighting misses some math-heavy slides (4 of 151 real chunks) and marks whole lines for right-to-left matches.
  - The privacy badge covers the app's process tree only; Ollama is a separate process.

- **Parsing:** code fences are tracked per PDF page; setext (`===`/`---`) headings aren't recognized; sentence splitting doesn't know abbreviations; PDF headings are only detected when their font is larger. UTF-8 only, no OCR (Phase 6).
- **Index:** exact vector scan with no ANN index (fine at this scale). `add` writes parents and children in two steps, not one transaction.
- **FTS tokenizer:** LanceDB tokenizers that load language dictionaries (jieba, lindera) may **download** them. We use the simple tokenizer with the built-in English stemmer; keep it that way.
- **Reranker load time:** `ask` loads the cross-encoder per invocation (~7 s). The UI (Phase 5) should keep it loaded.
- **Eval:** ranks aren't deduplicated before the cut at k. Every JSONL line must include `source_file` and `source_page` keys (they can be `null`).
- **Agent:**
  - The reranker scores passages against the standalone question only, so English questions about Hebrew text and two-part questions can fail grading (in the personal set, the right passage scored 0.05–0.12).
  - The direct (no-retrieval) path relies on the analyze call's `needs_retrieval`. The direct reply refuses factual questions, so a misclassified question gets "not found" instead of a search.
  - `trace RUN_ID` needs the full id, and stored runs are never pruned.
- **Hebrew parsing:** the bidi pass is a heuristic.
  - Formula-only lines on Hebrew pages are read left to right.
  - Side-by-side text boxes on one baseline merge into one row.
  - Some math subscripts become `?`.
  - "Hebrew, number, English" sequences can come out in visual order.
- **Baseline:** abstention is detected by a prefix match on `NOT_FOUND`. If the model rephrases it, it counts as an answer: 2 of 60 CUAD answers in the baseline run. A file whose H1 equals its file name gets a label like "CLAUDE.md, CLAUDE.md".
- Embedding batch size (32) isn't in the config yet.
- `vault/README.md` gets indexed like any document, so it shows up as a (harmless) distractor. Consider skipping it in ingest.
- CUAD contract names come from EDGAR filer names (`Aspirityholdingsllc Outsourcing Agreement (2012)`), which often differ from the party names inside the contract. Questions find the right contract mostly through the title prefix on every chunk. That's realistic (users refer to files by name), but keep it in mind when reading results.

## Environment gotchas (Windows dev machine)

- **Console code page is cp1255.** Keep CLI output ASCII (`[ OK ]`/`[FAIL]`, not ✓/✗). `cli.py` sets `errors="replace"` as a safety net.
- **`ollama` may not be on PATH** in a new shell. The executable is `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`.
- **Line endings:** `core.autocrlf=true`, so the "LF will be replaced by CRLF" warnings are harmless.
- **RTX 50 series (Blackwell):** torch comes from the `cu128` index (`[tool.uv.sources]` in `pyproject.toml`), only with `uv sync --extra rerank`. CI installs without the extra; mypy treats torch/sentence-transformers as untyped there.
- **VRAM:** `qwen3:14b` (11 GB at `num_ctx` 16384) + the embedder (2.4 GB) + the reranker (1.1 GB) fit in 16 GB, all on GPU.
- Live tests (`-m ollama`) are excluded from default `pytest` runs and from CI.

## How the work was split (for reuse)

Phase 1 was built by four parallel agents, each in its own git worktree. The interfaces were fixed in stubs first so the merges were conflict-free; an integration pass then wired everything together.

- Task prompts: `scripts/agents/prompts/` (`_rules.md` holds the shared rules).
- In this session the agents were run with Claude Code's built-in subagents (`isolation: worktree`). The PowerShell scripts in `scripts/agents/` (`setup.ps1` → `start.ps1` → `merge.ps1`) do the same thing by hand. They pass a syntax check but **have not been run end to end yet**.
- To reuse for a later phase: write new stubs and contracts first, add one prompt per agent, and update `$Agents` in `scripts/agents/_common.ps1`.
- Phase 2 used three agents (runner, judge, adversarial); the shared metrics and CLI were done on `main` before and after. Phase 4 used four (graph, safety, traces, personal documents) after the contracts commit `5051fb2`; all of them merged without conflicts. **Gotcha:** Claude Code's worktree agents branched from the commit the session started on, not current `main`, so each had to `git merge --ff-only main` to see the contracts. Put that step in the launch prompt.
