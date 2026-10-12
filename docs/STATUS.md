# VaultMind: Status and Handoff

_Last updated: 2026-10-13 (Phase 5.5 M0 and M1 on `main`; M2, the planner and recipes, is next)._

This is the starting point for anyone (human or agent) continuing the work. Read it after `CLAUDE.md`. The full roadmap with checkboxes is in [EXECUTION_PLAN.md](EXECUTION_PLAN.md).

## Where things stand

| Phase | State |
|---|---|
| 0. Project setup | ✅ Done. Eval corpus: 25 CUAD contracts + 10 adversarial documents. Your own mixed-type documents still optional |
| 1. Ingestion + baseline RAG | ✅ Done and verified against real models |
| 2. Evaluation harness | ✅ Done. Baseline in [RESULTS.md](RESULTS.md), reproducible. `eval generate` deferred |
| 3. Retrieval quality | ✅ Done. Hybrid + rerank + child chunks are the new default; ablations in [RESULTS.md](RESULTS.md) |
| 4. Agent loop + safety | ✅ Done. LangGraph loop, injection guard, contradiction and groundedness checks, traces. Measured in [RESULTS.md](RESULTS.md); only the parts that earned their keep are on by default |
| 5. Desktop app | ✅ Built (NiceGUI, `vaultmind ui`). Opens as a browser tab by default. Hand walkthrough partly done |
| 5.5 Document agent | 🟡 **In progress.** M0 (contracts) and M1 (document cards, tools + gateway, task eval) done. **M2 next:** planner + recipes. Until M2 lands, the app still only looks things up (see below) |
| 6–7 | ⬜ Not started |

## Where we stopped (2026-10-13)

**Done 2026-10-12 evening to 2026-10-13:** the three Phase 5.5 decisions were made; M0 (shared contracts) landed on `main`; M1 was built by three parallel agents (document cards, tools + gateway, task eval), reviewed, merged and wired into the app; cards were built for the whole real vault. 837 unit tests, ruff and strict mypy pass. Everything is pushed.

**Real vault, cards built (2026-10-13):** `vaultmind enrich` wrote 47 cards with 0 failures in 10 min 15 s (~13 s per document with the GPU to itself), 272 section summaries with none empty, 39 English and 8 Hebrew. The handwritten Hebrew PDF is still rejected at ingest (no usable text layer), as before.

**What the app can do today:** still only lookups. The cards, catalog and tools exist and are tested, but nothing routes a message to them yet: that is M2. Run `uv run vaultmind ui` to see the app as it was; run `uv run vaultmind enrich` after adding files without the app.

**Next session, start here:** M2 (planner + recipes + transform) in EXECUTION_PLAN.md. In order:
1. Planner node: one schema call with `catalog.describe(agent.catalog_files)`, the chat history and the request, returning a `Plan`; mentions resolved with `Catalog.resolve`, `AmbiguousDocument` → a clarifying question (M3 makes it an interrupt; in M2 it can simply be the answer).
2. Route in the graph: `lookup` → today's graph unchanged; `chat` → direct; `transform` → rework the previous answer; `list | summarize | compare | extract | generate` → recipes over the gateway; `open` → for now, the closest recipe (the loop is M3). `agent.router: false` keeps today's behavior for the ablation.
3. Recipes cite the sections a tool read (the decided citation rule); answers name the documents they read.
4. A task runner in the eval (planner + recipes over `personal_tasks.jsonl`): router accuracy, task success with the key-facts judge, latency per task type; CUAD and personal lookups must hold within noise.
5. Done when the four failing requests below (and their Hebrew versions) get correct, cited answers.

Before M2's eval numbers mean anything, **you** check `eval/datasets/personal_tasks_review.md` by hand.

## Earlier: end of day, 2026-10-12

**Done today:** the 27 personal reference answers were verified (none changed); Phase 5, the app, was built by 5 parallel agents in two waves, merged, and run on the real vault with the real models; the first hand test led to fixes (layout, greetings); two security problems were found and fixed (LAN exposure, DNS rebinding / cross-site sockets); browser mode became the default. 639 unit tests, ruff and strict mypy pass.

**Phase 5 state:** `uv run vaultmind ui` (a tab in your default browser) or `vaultmind ui --native` (own window). Verified live: the engine loads in ~15 s plus ~10 s Ollama warm-up (a loading screen covers it), first token ~2.9 s, follow-ups ~1.6 s; the startup scan of an unchanged vault takes ~2.6 s; 12 questions asked while 9 files re-indexed in a loop all answered correctly. Served on 127.0.0.1 only, refuses other websites (Host/Origin check), no external assets. Layout checked with headless Edge screenshots at 600 and 1280 px.

**Decided: browser mode is the default** (`ui.native: false`). In native mode the window's WebView2 (Microsoft Edge runtime) opens HTTPS connections to Microsoft servers by itself; switching off its background services and a dead local proxy removed the network-service traffic, but its main process still connects through Windows' own networking. In browser mode VaultMind's process tree is one Python process with no outside connections (verified; the badge is green). The trade-off: the browser can see the page, so **use a separate browser profile with no extensions** for VaultMind, and don't use "Translate this page", AI sidebars or enhanced spell check on its tab. `--native` stays available.

### Major gap found by the user: it searches, it doesn't act like an agent

Asking it to summarize, list or compare fails. Live check with the real models (2026-10-12):

| Request | Result |
|---|---|
| "Summarize the Workato Recipe Design Worksheet" | "I couldn't find this in your documents." |
| "Summarize the AI in the workplace syllabus" | A "summary" of the title page only (it read one 256-token chunk) |
| "What documents do I have?" | "I couldn't find this in your documents." |
| "Compare the Solution Design Checklist with the Recipe Design Worksheet" | "I couldn't find this in your documents." |

**Why:** every message takes the same path: search, keep the top 5 small passages that pass the reranker threshold, answer only from those. That fits lookups ("what is the deadline"), and nothing else:
- a summary needs a whole document, not 5 fragments;
- "what documents do I have" needs the file registry, which the agent can't see;
- a comparison needs two whole documents;
- a request like "summarize X" scores low against any single passage, so grading drops it to "not found".

The "agent" today is a corrective retrieval loop (rewrite, check, cite), and Phase 4 showed even its retrieval steps add little. It has no tools and doesn't decide what kind of task it was given. **Fixing this is the next phase (5.5 in EXECUTION_PLAN.md) and comes before hardening:** it's what the product is supposed to be.

**The plan (2026-10-12):** turn it into a local document agent that works like a cloud assistant over your files: a ReAct loop with human-in-the-loop (plan, reason, call tools, observe, ask when unsure), but **behind fast paths**. A planner call classifies every message (`chat | transform | lookup | list | summarize | compare | extract | generate | open`):
- lookups keep today's measured corrective-RAG graph (~1.5 s, ~0.80 CUAD correctness);
- common tasks run as fixed tool sequences (recipes), which a 14B model handles reliably;
- only open, multi-step tasks enter the tool-calling loop (qwen3 native tool calls, thinking on, bounded steps).

The biggest lever is a **document card** per file, written at ingest (type, language, summary, outline, key entities and dates, per-section summaries): it lets the agent know the library, resolve "the syllabus" to a file, and answer whole-document questions without fitting the whole document into `num_ctx`. Tools are read-only; writes (save a note, export) need approval; the agent has no delete tool. Milestones M0–M5, the eval set and latency targets are in EXECUTION_PLAN.md.

### What's left, in order

1. **Phase 5.5: document agent** (plan in EXECUTION_PLAN.md). ~~Decisions~~, ~~M0~~, ~~M1~~ → **M2 planner + recipes** → M3 ReAct loop + human-in-the-loop → M4 retrieval upgrades → M5 UI.
   - **You:** check the 63 task items by hand in `eval/datasets/personal_tasks_review.md` (gitignored), as you did for `personal.jsonl`. Facts that bundle two details should be split: the key-facts judge is lenient on them.
2. **Finish the hand walkthrough** of the app: drop a PDF in, ask, open a citation, open the trace, export, change a setting. Only the chat page has been tried by hand so far.
3. **Phase 6: hardening.** Network-blocked test, telemetry audit, launch token (other local programs can use the app today), atomic re-index, Ollama-down recovery, resumable ingestion, edge-case files, secret/PII scanner. OCR needs a decision (new dependency + model download; offline Hebrew handwriting OCR is weak).
4. **Phase 7: packaging and showcase.** One-command install, demo GIF, results write-up, CV bullets.
5. **Quality items, any time:** harder follow-up and injection items. (The generation-side refusal fix, multi-phrasing grading and the key-facts judge are now scheduled in Phase 5.5, M4 and M1.)

**Decided for 5.5 (2026-10-12):**
1. **Citations:** summaries, comparisons, extractions and generated material cite the sections and pages a tool read. Lookups keep the graded-passage rule and "not found"; a document that doesn't exist gets "not found". Now in `CLAUDE.md`.
2. **General knowledge:** strict "only my documents" by default; `agent.world_knowledge` (off) allows clearly labeled answers from the local model.
3. **Document cards at ingest:** on (`ingest.enrich: true`), in the background, once per file content.

**M1, merged 2026-10-13** (three agents in parallel, then integrated by hand; 837 unit tests, ruff and strict mypy pass):
- **Document cards** (`ingest/enrich.py`): section summaries in batches that fit half of `num_ctx`, then one card call. The language is detected from the script (Hebrew if ≥30% of letters) and named in the prompt: left to itself, qwen3 wrote Hebrew summaries of English contracts, or English summaries of Hebrew lectures. Live: 21–47 s per document (1.8–5 s per section) with the GPU shared; the whole real vault took 10 min for 47 cards. `Enricher` runs in the app with ingestion, gets every file that becomes ready, and pauses while a question runs. `vaultmind enrich` builds the missing cards from the command line.
- **Catalog, tools, gateway** (`agent/catalog.py`, `agent/tools.py`, `agent/gateway.py`): 9 read-only tools (list_documents, document_card, outline, read, search with an optional document scope, summarize with cited sections, extract, calc, date_diff). Resolution: exact id/name/title → fuzzy words → the reranker (multilingual; right documents scored 0.49–0.99, wrong ones ≤0.05 on 8 synthetic mentions); close calls raise `AmbiguousDocument`. Live: qwen3:14b made a valid first tool call on **10/10** requests (median 2.1 s thinking off, 8.5 s on). The ollama client silently drops `anyOf`/`$ref`/`default` from tool schemas, so specs are flattened (a test checks them). `calc` has an input-length cap (a 1500-term sum hit the recursion limit, found in review).
- **Task eval** (`eval/judge.py`, `eval/metrics.py`, `eval run --repeat N`): key-facts judge (fixed-length list, the answer's own words as evidence; 29/29 facts on the tuning cases, 8/9 held out, stable over 3 runs), clarify judge (4/4), router accuracy/confusion, task success, mean (min–max) over repeats. The task set: 63 items, 31 in Hebrew, in the gitignored `personal_tasks.jsonl`; synthetic format example in `eval/datasets/tasks_example.jsonl`.
- **Open for M2+:** scoped search over-fetches (10× candidates, max 200) and filters by file; a `file_id` prefilter in `ChunkIndex.vector_search` / `fts_search` would be exact (contract change request). The summarize cache is in memory only. Latin-script documents count as English for cards.

**M0, what's on `main` now (2026-10-12):**
- `LLM.chat(messages, tools=, think=) -> AssistantTurn` with `ToolSpec` / `ToolCall` and a `tool` message role (`core/providers.py`); Ollama implementation; `FakeLLM(turns=...)` and `tool_turn()` play back scripted tool calls. Live with `qwen3:14b`: "What documents do I have?" → a correct `list_documents` call in 0.3 s (thinking off) or 2.1 s (thinking on, ~700 chars of reasoning); given the tool result, it answered from it in 0.4 s.
- `DocCard` (`core/models.py`) and `CardStore` (`store/cards.py`, registry migration 4), keyed by (sha256, model, prompt version); `ChunkIndex.parents_for_file`.
- `TaskKind`, `Plan`, `ToolResult`, the tool exceptions and the trace keys for `plan`, `recipe`, `reason`, `tool`, `human`, `respond` (`agent/types.py`).
- The `tasks.jsonl` format (`eval/tasks.py`).
- Config: `ingest.enrich`, `agent.router`, `agent.world_knowledge`, `agent.catalog_files`, `agent.max_steps`, `agent.observation_tokens`, `agent.think_open`. **They do nothing yet**; each gets wired in the milestone that uses it, and they're left out of the Settings page until then.
- Stubs for M1: `ingest/enrich.py`, `agent/catalog.py`, `agent/tools.py`, `agent/gateway.py`. Dependency `langgraph-checkpoint-sqlite` (for M3; no network code in it, checked).
- Integration after M1 (by hand, on `main`): start the `Enricher` in `Engine` (wrap `ask` in `paused()`, chain the queue's `ready` events to `submit`), add a `vaultmind enrich` command, build cards for the real vault.

**To resume:** `uv run vaultmind doctor`, then `uv run vaultmind ui`. Ollama must be running (`%LOCALAPPDATA%\Programs\Ollama\ollama.exe` if it isn't on PATH).

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
| **Other websites could read and drive the app.** 127.0.0.1 keeps the network out, but not a malicious page in a local browser: the server answered a forged Host header (DNS rebinding) with the full library page, and NiceGUI's socket.io allows any origin. Affects native and browser mode alike | `ui/local_only.py`: ASGI middleware that returns 403 unless Host is our address and Origin (when sent) is our page. Verified live: forged Host and foreign-origin socket.io/websocket requests get 403, our own page works |
| First hand test: "HI" got "not found" plus three unrelated closest matches. Analyze is skipped on a chat's first message, so nothing classified it as a greeting | `agent.smalltalk` (on): when nothing passes grading on an unclassified message, one short direct reply; factual questions still get not-found with the closest matches. Live: greetings (English, Hebrew) answered, "capital of Peru" and Bellman-Ford stay not-found |
| First hand test: the chat column was ~230 px wide, the answer and chips cut off, the user's own message invisible | NiceGUI's sub-pages container shrink-wraps its page; scroll-area content could outgrow the window; a Tailwind color that doesn't exist. Fixed and checked with headless Edge screenshots at 600 and 1280 px |
| Merged UI tests waited for "any chip" to mean "the answer arrived", but the header's privacy badge is a chip too | Tests use the citation chips' marker |

## Redesign considerations (each needs a measurement before it lands)

Ideas 1–4 are now scheduled in Phase 5.5 (1–2 in M4, 3 in the planner work, 4 in M1); idea 5 is superseded by the document-agent plan.

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
5. **Keep LangGraph.** With variants and rewrites off, today's graph is nearly linear, but it pays for itself in streaming, per-node traces and the switches the eval needs. Phase 5.5 grows it instead of replacing it: a planner/router in front, the current lookup graph reused as one path, a tool loop with `interrupt()` for human-in-the-loop and a local SQLite checkpointer.

**Eval corpus (2026-10-12):** CUAD's `data.zip` is copied to `eval/corpora/cuad/` (gitignored, 18 MB, from the `../cuad` repo). `uv run vaultmind eval import-cuad` writes 25 contracts to `vault/cuad/` and `eval/datasets/cuad.jsonl`; the same seed always gives the same subset. Ingest takes about 20 s. All 12 unanswerable items were checked by hand against the contract text (two bad labels found and fixed via `RELATED` categories in `eval/cuad.py`). The full CUAD release with the 510 original PDFs is at `eval/corpora/cuad/CUAD_v1.zip` (md5 verified) for the optional txt-vs-PDF comparison. CUAD is CC BY 4.0: credit it (Hendrycks et al., NeurIPS 2021, The Atticus Project) in `RESULTS.md` and the README.

## What works today

```powershell
uv sync --extra rerank                  # torch cu128 + sentence-transformers
uv run vaultmind pull-reranker          # one-time download (the only Hugging Face network step)
uv run vaultmind doctor                 # Ollama reachable, both models pulled, reranker cached
uv run vaultmind ingest [PATH]          # index vault/ (or PATH); incremental
uv run vaultmind enrich                 # write missing document cards (the app does this by itself)
uv run vaultmind ask "question" [--trace]   # agent loop: streamed answer, [n] citations, warnings, stored trace
uv run vaultmind trace [RUN_ID] [--last N]  # stored runs and their traces
uv run vaultmind config                 # effective config
uv run vaultmind eval import-cuad       # CUAD subset -> vault/cuad/ + eval/datasets/cuad.jsonl
uv run vaultmind eval install-adversarial  # poisoned + contradicting docs -> vault/adversarial/
uv run vaultmind eval run -d cuad -d adversarial -d personal [-c a.yaml -c b.yaml] [--no-judge] [--limit N] [--repeat N]
uv run vaultmind eval sweep eval/results/<run>/<config>__cuad.jsonl -d cuad   # threshold replay
```

Verified on the dev machine (Windows 11, RTX 5070 Ti 16 GB, Ollama with `qwen3:14b` + `qwen3-embedding:0.6b`):

- 837 unit tests pass (fakes, no GPU). Live tests run with `uv run pytest -m ollama` (they load the reranker too).
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
| `eval/judge.py` | LLM judge: one schema call, verdict correct / partial / incorrect + reason, one retry, `JudgeError`. Phase 5.5: `judge_key_facts` (per-fact found + evidence) and `judge_clarify` | Prompts in module constants. Correctness judge: 11 of 15 hand-checked verdicts agreed. Key-facts judge: 29/29 tuning facts, 8/9 held out |
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
| `store/cards.py` | `CardStore`: document cards in `vaultmind.sqlite` (migration 4), keyed by (sha256, model, prompt version); `latest`, `prune` | Phase 5.5 M0 |
| `ingest/enrich.py` | `build_card` (batched section summaries, then one card call; language detected from the script), `Enricher` (background worker, `paused()` while a question runs) | Started by `Engine.start_ingest` when `ingest.enrich`; `vaultmind enrich` from the CLI |
| `agent/catalog.py` | `Catalog`: ready files + cards; `resolve(mention)` (id → name/title → fuzzy → reranker; `AmbiguousDocument` / `DocumentNotFound`); `describe()` for the planner | Phase 5.5 M1 |
| `agent/tools.py`, `agent/gateway.py` | 9 read-only tools (list_documents, document_card, outline, read, search, summarize, extract, calc, date_diff); `Gateway`: argument validation, step budget, repeat cache, observation budget, injection guard, approval for writes | Not yet used by the graph (M2). No write or delete tools |
| `eval/tasks.py` | `TaskItem`, `load_tasks`, `save_tasks`: the task eval format | Personal set (63 items) is gitignored; `eval/datasets/tasks_example.jsonl` is synthetic |
| `agent/trace_text.py` | One-line trace summaries shared by the CLI and the app | |
| `cli.py` | `config`, `doctor`, `pull-reranker`, `ingest`, `ui [--native/--browser]`, `ask [--trace]`, `trace`, `eval import-cuad`, `eval install-adversarial`, `eval run`, `eval sweep` | Forces `errors="replace"` on stdout (Windows cp1255 console) |

Storage lives in `paths.data_dir` (default `./data`): `index/` (LanceDB) and `vaultmind.sqlite`. Both are gitignored.

## Decisions already made (don't re-litigate without measurements)

- **No API layer.** The UI (NiceGUI) calls the engine in-process. **Browser tab by default, not the native window** (2026-10-12: WebView2 contacts Microsoft by itself). The engine never imports `ui/`.
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

### Phase 5: Desktop app (built 2026-10-12)
`vaultmind ui`, a NiceGUI native window with:
- streamed chat with citation chips, and a trace panel
- a library with per-file status, and live folder watching
- settings
- highlighted citations (PDF page + passage)
- export to Markdown/PDF
- the "localhost only" privacy badge

Load the models once at startup.

### Phase 5.5: Document agent (next)
ReAct with human-in-the-loop behind fast paths: document cards at ingest, working memory per chat, a planner/router, tools behind a validating gateway (list, card, outline, read, scoped search, summarize, extract, calc; writes need approval; no delete), recipes for list/summarize/compare/extract/generate, an open tool loop for multi-step tasks, clarify/approve via `interrupt()`, retrieval upgrades (multi-phrasing grading, query translation, refusal second pass), a live step list in the UI, and ~60 new eval items (half Hebrew). Milestones M0–M5 in EXECUTION_PLAN.md. Needs the three decisions above.

### Phase 6: Hardening
- Launch token, so other programs on the machine can't use the app (other websites are already refused).
- Atomic re-index (a question asked mid-re-index can miss that file).
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
