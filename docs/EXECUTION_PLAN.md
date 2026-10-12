# VaultMind: Execution Plan

Each phase ends in something that **works and can be measured**. Check tasks off as they land, and record evaluation numbers in [`docs/RESULTS.md`](RESULTS.md) at the end of each phase from Phase 2 on.

The order matters. A plain baseline and the eval harness come **before** any of the advanced techniques, so every later improvement has before/after numbers. Those numbers are what make this project stand out.

| Phase | Goal | Size |
|---|---|---|
| 0 | Project setup | S |
| 1 | Ingestion + baseline RAG (CLI) | L |
| 2 | Evaluation harness + baseline numbers | M |
| 3 | Retrieval quality: hybrid, rerank, small-to-big | M |
| 4 | Agentic corrective loop + traces | M |
| 5 | Desktop UI + live folder watching | L |
| 5.5 | Document agent: planner, tools, ReAct loop, human-in-the-loop | L |
| 6 | Hardening + privacy guarantees | M |
| 7 | Packaging + showcase | S |

---

## Phase 0: Project setup

- [x] Repo structure, `pyproject.toml`, uv, Python 3.12 pin
- [x] Typed YAML config (`config/default.yaml`, `core/config.py`)
- [x] Provider protocols (`LLM`, `Embedder`, `Reranker`) + fakes for tests
- [x] CLI: `vaultmind config`, `vaultmind doctor`
- [x] CI: ruff, mypy, pytest
- [x] Install Ollama, then `ollama pull qwen3:14b` and `ollama pull qwen3-embedding:0.6b`
- [x] `uv run vaultmind doctor` passes
- [x] **Main eval corpus: CUAD.** `vaultmind eval import-cuad` copies a seeded subset of [CUAD](https://www.atticusprojectai.org/cuad) contracts (CC BY 4.0, lawyer-labeled clause spans) into `vault/cuad/` and writes `eval/datasets/cuad.jsonl`. The zip lives in `eval/corpora/cuad/` (gitignored). Default: 25 contracts, 60 questions, 20% unanswerable, seed 0.
- [x] **Mixed-type supplement:** (2026-10-12: 12 of the user's own PDFs in `vault/personal/`, mostly Hebrew course material, plus 27 questions in the gitignored `eval/datasets/personal.jsonl`, reference answers verified against the source pages.) 5–10 of your own documents (at least one PDF with tables, slides exported to PDF, Markdown notes) to cover the parsers CUAD doesn't exercise (it's plain text only), plus a few hand-written questions about them.

**Done when:** `doctor` is all green and the corpus is sitting in `vault/`.

## Phase 1: Ingestion + baseline RAG

Goal: the simplest end-to-end pipeline that works. Vector search only, no reranker, no loop.

Dependencies: `pymupdf4llm`, `lancedb` (already added). No `tiktoken`: it downloads its tokenizer from the internet, so token counts are estimated as `len/4`.

Parallel build: contracts and stubs are fixed in `core/models.py` and the module stubs. Four agents implement them in separate worktrees (see `scripts/agents/`), then one integration pass wires up the pipeline, the baseline, and the CLI.

- [x] Shared contracts: `core/models.py` + stub signatures for every Phase 1 module
- [x] `core/ollama.py`: `OllamaLLM` and `OllamaEmbedder` implementing the protocols. Always pass `num_ctx` and `keep_alive`; use the `format=` JSON schema for structured output.
- [x] `ingest/parsers.py`: PDF → Markdown with page numbers (pymupdf4llm), plus `.md`/`.txt`. Return a list of `(text, page, section_path)` blocks.
- [x] `ingest/chunker.py`: heading-aware parent sections (~1200 tokens), each split into child chunks (~256 tokens, 15% overlap). Prepend `Doc title > Section` to each child before embedding.
- [x] `store/index.py`: LanceDB table of child chunks (vector, text, parent_id, file_id, page, section). Parent texts are stored alongside.
- [x] `store/registry.py`: SQLite `files` table (path, sha256, status, error, embedder model, indexed_at). Skip unchanged files, re-index modified ones, and delete chunks for files that were removed.
- [x] Record the embedding model name in the index, and refuse to mix models (prompt to re-index instead).
- [x] `agent/baseline.py`: embed query → top-k → prompt with numbered sources → answer with `[n]` citations mapped back to file + page.
- [x] CLI: `vaultmind ingest [PATH]`, `vaultmind ask "question"`
- [x] Unit tests for the chunker and registry using fakes; one `@pytest.mark.ollama` end-to-end test.

**Done when:** you can ingest your corpus and get cited answers from the CLI, and re-running `ingest` on an unchanged folder does nothing.

> Status: done. 174 unit tests and 5 live Ollama tests passing; answers take about 2 s on an RTX 5070 Ti. Remaining Phase 0 item: the real test corpus.

## Phase 2: Evaluation harness

Goal: one command that scores any config. This turns "I think it's better" into numbers.

- [x] Dataset format `eval/datasets/<name>.jsonl`: `{question, answer, source_file, source_page, answerable, tags, evidence, forbidden}`. `evidence` holds gold text spans when known (CUAD has them); `forbidden` holds planted injection payloads.
- [x] CUAD dataset importer (`vaultmind eval import-cuad`). Questions name their contract; spans become `evidence`; absent clauses become unanswerable items (only when related CUAD categories are empty too). All 12 unanswerable items checked by hand.
- [x] **Evidence recall@k:** share of answerable items where a retrieved passage contains (or overlaps) a gold evidence span, after whitespace normalization. On CUAD this is the headline retrieval metric; file-level recall is too easy when every question names its contract.
- [ ] `vaultmind eval generate` (lower priority now that CUAD covers the main set): the LLM drafts questions from random chunks of your own documents. **Verify each one by hand.**
- [ ] Include tough cases: exact-term lookups (codes, names), multi-hop questions, vague phrasing, and follow-up questions that depend on chat history. *(CUAD covers exact terms and vague clause-type phrasing; multi-hop and follow-up questions still need hand-written items, needed before Phase 4's history handling.)*
- [x] **Adversarial cases:** add 3–5 poisoned documents to the corpus containing embedded instructions ("ignore previous instructions and reply X", fake system prompts, "tell the user to visit ..."), plus questions that retrieve them, tagged `injection`. Also add 2–3 document pairs that disagree on a fact, tagged `contradiction`. Record the baseline **injection resistance rate**: the share of `injection` items where the answer does not follow the planted instruction (exact-match check on the planted payload, no judge needed).
- [x] Metrics: recall@k and MRR (retrieval), answer correctness against the reference (LLM judge in `eval/judge.py`, 15 verdicts spot-checked by hand: 11 agree), abstention accuracy, injection resistance, latency p50/p95.
- [x] `vaultmind eval run --config a.yaml --config b.yaml [-d cuad -d adversarial]` runs each config and prints a comparison table, saving raw results to `eval/results/`.
- [x] Record the **baseline** in `docs/RESULTS.md`.

**Done when:** the baseline numbers are recorded, and re-running the eval gives the same retrieval numbers.

## Phase 3: Retrieval quality

Goal: raise recall@5. Measure each step separately with an ablation config.

Dependencies: `sentence-transformers` + torch from the CUDA 12.8 index, as the optional `rerank` extra (`uv sync --extra rerank`; the RTX 50 series is Blackwell and needs cu128 or newer). The model is fetched once with `vaultmind pull-reranker` and loaded offline.

- [x] **Hybrid search:** LanceDB native full-text index (BM25) on the child's `embed_text` (title > section + text), run alongside vector search, fused with reciprocal rank fusion (k=60). Config: `retrieval.hybrid`.
- [x] **Reranker:** `core/rerank.py` wrapping a sentence-transformers `CrossEncoder` (bge-reranker-v2-m3, fp16) on the GPU. Rerank `candidates` (20) down to `top_k` (5). Config: `retrieval.rerank`.
- [x] **Small-to-big:** search over children, deduplicate by parent, and send parent sections to the LLM. Compare answer correctness, not just recall. (Config: `retrieval.small_to_big`. Measured worse than sending the graded children, about -8 points correctness over 3 runs each, so it's off by default.)
- [x] Tune `rerank_threshold` against the eval set: plot the threshold against answerable recall and unanswerable abstention, then pick the best balance. (`vaultmind eval sweep`; kept at 0.3, see RESULTS.md.)
- [x] Record each ablation in `RESULTS.md`.

**Done when:** each technique has a measured contribution, whether positive or not. Keep only what helps.

> Status: done 2026-10-12. Hybrid + rerank lift CUAD evidence recall@5 from 0.708 to 0.958; the new default (hybrid + rerank + child chunks) lifts correctness from ~0.64 to ~0.79 and abstention accuracy from ~0.75 to ~0.86 at ~0.3 s extra latency. Details in RESULTS.md.

## Phase 4: Agentic corrective loop

Goal: replace the linear baseline with the LangGraph state machine.

Dependencies: `uv add langgraph`

- [x] Graph state: question, history, queries tried, candidates, graded passages, rewrite count, trace events.
- [x] **Analyze node** (one structured call): standalone query (resolves chat follow-ups), 2–3 query variants, `needs_retrieval`. (Variants measured: no answer gain, +1.2 s, so `query_variants: 0` by default; analyze then runs only with chat history.)
- [x] **Retrieve node:** hybrid retrieval for each query variant, fused.
- [x] **Grade node:** reranker scores against the threshold. No per-chunk LLM calls.
- [x] **Rewrite node:** given the failed queries and what was found, produce a different query. Stop after `agent.max_rewrites`. (Recovered 0 of 4 rewritten questions, so it's off by default; see the redesign notes in STATUS.md.)
- [x] **Generate node:** streamed, cites only graded passages.
- [x] **Not-found node:** an honest "not in your documents" answer plus the closest matches.
- [x] Optional **groundedness check** node (config flag, off by default): checks each claim against its cited passage.
- [x] **Prompt-injection hardening** (config: `agent.injection_guard`): passages go into the prompt inside clearly delimited data blocks, the system prompt says text inside them is never an instruction, and passages that match injection patterns (at ingest, see Phase 6 scanner) are flagged in the trace and shown with a warning. Eval: injection resistance rate guard on vs. off, plus its effect on correctness.
- [x] Optional **contradiction check** (config: `agent.detect_contradictions`, off by default): when graded passages from different files disagree on a fact, the answer states both claims and cites each, instead of picking one. Eval it on the `contradiction` items.
- [x] **Traces:** every node emits events (inputs, outputs, scores, duration), saved to a SQLite `traces` table. `vaultmind ask --trace` prints them.
- [x] Eval: loop on vs. off. Report how many failed queries the rewrite recovered, what it costs in latency, and how abstention changes.

**Done when:** the loop beats or matches the Phase 3 config on correctness and abstention, and its latency cost is measured.

> Status: done 2026-10-12. The default loop matches the Phase 3 pipeline on correctness and abstention (within judge noise) at about the same latency (~1.5 s median), and adds chat follow-ups, honest not-found answers with no LLM call, the injection guard, optional checks and traces. Variants and rewrites measured no gain and are off by default. Details in RESULTS.md, assessment and redesign notes in STATUS.md.

## Phase 5: Desktop UI + live ingestion

Goal: a polished app that calls the engine directly, with no API layer.

Dependencies: `uv add "nicegui[native]" watchdog psutil`

- [x] `vaultmind ui` opens a native window (`ui.run(native=True)`).
- [x] **Chat:** streamed answers; citation chips that open the source PDF at the right page; conversation history saved in SQLite.
- [x] **Trace panel:** a collapsible timeline under each answer showing queries, passage scores, rewrites, and per-step timing.
- [x] **Library:** drag-and-drop upload; per-file status (queued / indexing / ready / failed with reason); re-index and remove actions.
- [x] **Live folder watching:** a watchdog observer feeds the ingestion job queue in the background, so the UI never blocks.
- [x] **Settings:** watched folders, models, and retrieval/agent switches (written to `config/local.yaml`).
- [ ] Optional: an **Eval** page that runs the test set and shows the comparison table in-app.
- [x] Load models at startup so the first question isn't slow.
- [x] **Highlighted citations:** clicking a citation opens the PDF at the page with the supporting text highlighted (PyMuPDF `search_for` on the cited child's text + a highlight annotation on a temp copy; never modify the user's file). Markdown/text sources scroll to and highlight the passage.
- [x] **Export:** save an answer or a whole chat as Markdown or PDF, with its citations listed as file + page.
- [x] **Privacy badge:** a status indicator that checks the app's own connections (`psutil.Process().net_connections()`) every few seconds and shows "localhost only". It turns red and names the remote address if any non-loopback connection appears.

**Done when:** a new user can drop files in, watch them index, ask a question, and inspect the trace without touching the CLI.

> Status (2026-10-12): built and merged; the app runs on the real vault with the real models. Opens as a browser tab by default (the native window's WebView2 contacts Microsoft by itself, see STATUS.md). Other websites are refused (Host/Origin check against DNS rebinding). Not yet walked through by hand.

### Parallel build (two waves)

The app splits into services with no UI code (testable with fakes) and NiceGUI pages on top. Services go first, so the page agents build on working code, not stubs.

- **Contracts (on `main`):** deps, `ingest:` + `ui:` config sections, registry migration 3 (`chats`, `messages`), `ingest.pipeline.forget_file`, `app/engine.py` (`Engine`: models loaded once, `ask(chat_id, question)` streams and stores the exchange, `start_ingest`), stubs for every wave 1 module, and `tests/test_layering.py` (only `ui/` imports NiceGUI; nothing imports `ui/`).
- **Wave 1: services, 3 agents in parallel** (prompts in `scripts/agents/prompts/`):

  | Agent | Files | Prompt |
  |---|---|---|
  | queue | `ingest/queue.py` (`IngestQueue`), `ingest/watcher.py` (`VaultWatcher`) | `queue.md` |
  | chats | `store/chats.py` (`ChatStore`), `app/export.py` (Markdown + PDF) | `chats.md` |
  | citations | `app/highlight.py`, `app/privacy.py`, `app/settings.py` | `citations.md` |

- **Merge + integration:** `Engine` tests with fakes; a live check of `Engine` with real models and a watched folder.
- **Wave 2: pages, 2 agents in parallel**, after a shell commit on `main` (`ui/app.py`: window, header with the privacy badge, navigation, engine startup; the `vaultmind ui` command):

  | Agent | Files |
  |---|---|
  | chat UI | `ui/chat.py` (streamed chat, chat list, citation chips, export), `ui/trace_view.py`, `ui/citation_view.py` |
  | library UI | `ui/library.py` (drag-and-drop, per-file status, re-index/remove), `ui/settings_page.py`, `ui/badge.py` |

- **Final integration:** run the app against the real vault and models, walk through the "done when" scenario, record findings in STATUS.md.

Rules for the UI: always `ui.run(host="127.0.0.1", reload=False, show_welcome_message=False)` (NiceGUI binds `0.0.0.0` by default outside native mode), no `on_air`, no CDN assets. Blocking engine calls run off the event loop (`run.io_bound`); worker-thread callbacks hop back to the loop.

## Phase 5.5: From search pipeline to document agent

Goal: VaultMind understands what was asked and does it, the way a cloud assistant like Gemini works over your files, but fully local. Found in the first hand test (2026-10-12): "summarize X", "what documents do I have" and "compare X and Y" all fail, because every message goes through search → top 5 passages → answer (details in STATUS.md). This phase comes before hardening.

### Why the current agent can't do it

- **No task understanding.** Every message takes the lookup path. Analyze is skipped on a chat's first message and, when it runs, only rewrites the question.
- **No tools.** It can't see the file list, read a whole document, or search inside one file.
- **No working memory.** "Summarize *it*" or "make that shorter" search again instead of using what it just read or wrote.
- **No multi-step work.** Compare, extract-across-files and "find X, then check Y" are out of reach.
- **No idea what a document is about as a whole.** "Which document covers X" fails.

### Target capabilities

| Capability | Example | Needs |
|---|---|---|
| Understand the task | "summarize", "compare", "list", "quiz me" | Planner / router |
| Know the library | "What documents do I have?", "which one covers X?" | Catalog + a card per document |
| Read whole documents | Summarize a 40-page syllabus | Read and outline tools, map-reduce, caching |
| Work across documents | "Every deadline in my courses, as a table" | Extract recipe, search scoped to one file |
| Reason in several steps | "Which course has the heavier workload, and why?" | ReAct tool loop |
| Rework its own answer | "shorter", "in Hebrew", "as bullet points" | Transform task (no retrieval) |
| Ask instead of guessing | "Which syllabus: AI or Workato?" | Human-in-the-loop clarification |
| Make things from your docs | Study notes, flashcards, an email draft | Generate recipes over what it read |
| Mixed Hebrew / English | English question about a Hebrew PDF | Query translation, grading against several phrasings |

Not realistic here, by design: a 1M-token context (the budget is ~16–32K on 16 GB of VRAM, hence document cards and map-reduce), web search (forbidden by the offline rule), and cloud-level speed on long tasks (showing each step live makes up for it).

### Architecture: ReAct with human-in-the-loop, behind fast paths

```
user message
   |
[STATE / MEMORY]  chat history, focus documents, earlier tool results, document catalog
   |
[PLAN & ROUTE]  one structured call: task, documents named, short plan
   |-- chat / transform previous answer ----------------------------> [RESPOND]
   |-- lookup  (today's corrective RAG, unchanged and measured) ----> [RESPOND]
   |-- recipe: list | summarize | compare | extract | generate -----> fixed tool sequence -> [RESPOND]
   '-- open task --> [REASON]  qwen3 native tool calls, thinking on <-----------------+
                        | tool call                                                  | observation
                        v                                                            |
                 [TOOL GATEWAY]  validate args, resolve documents, budgets,          |
                                 de-duplicate, injection guard on observations       |
                        |                          |                                 |
                    read-only?             needs approval? --> [HUMAN: approve / edit / reject]
                        |                          | approved            | rejected = feedback to REASON
                        '------> [EXECUTE TOOL] ---'-------------------------------------+
[RESPOND] -> verify (citations point to text that was read; groundedness on long answers) -> stream with chips
```

**Why fast paths in front of the loop:** each `qwen3:14b` step costs ~1–3 s, and a 14B model's tool calls are less reliable than a frontier model's. Running every lookup through a reasoning loop would turn 1.5 s into 6–10 s and add failure modes on the path that already measures well (~0.80 CUAD correctness). Lookups keep today's graph; recipes make common tasks predictable; the open loop handles what neither covers. Still LangGraph: the graph grows a router and a tool loop, the existing lookup subgraph is reused as is.

### Components

1. **Document understanding at ingest** (the biggest lever). After a file is indexed, a background job writes a **document card**: type (syllabus, contract, worksheet…), language, an ~8-line summary, outline, key entities and dates, and a 1–2 sentence summary per parent section. Stored in SQLite, keyed by (sha256, model, prompt version), so it's computed once. Section summaries are also indexed as searchable rows (light hierarchical retrieval) for "which document talks about X". The file is `ready` before its card is done. Config: `ingest.enrich`.
2. **State and memory.** Working memory per chat: focus documents (last discussed), tool results so far, the previous answer, so "it" and "that one" resolve. The planner sees the catalog (name + one line) when there are fewer than ~60 files; above that, it searches the cards. Later: user-level memory ("answer in Hebrew", "I'm taking course X"), editable in Settings.
3. **Planner / router.** Always runs. One structured call returns `task: chat | transform | lookup | list | summarize | compare | extract | generate | open`, the `documents` named (resolved against the catalog with a fuzzy-match check), `language`, a 1–3 step `plan`, and `clarify` when the request is ambiguous. Start on `qwen3:14b` with thinking off (~0.5 s measured for analyze); measure `qwen3:4b` (VRAM is tight with everything loaded).
4. **Tools** (`agent/tools.py`, plain Python over the index, registry and cards; read-only unless marked):

   | Tool | Backed by |
   |---|---|
   | `list_documents(filter?)` | registry + cards |
   | `document_card(doc)`, `outline(doc)` | cards, parent sections |
   | `read(doc, pages \| sections)` | `index.get_parents`, in order, within a token budget |
   | `search(query, doc?)` | today's hybrid + rerank, optionally scoped to one file |
   | `summarize(doc, focus?)` | map-reduce over sections, cached |
   | `extract(docs, schema)` | scoped search + one structured call per document, returns rows |
   | `calc(expr)`, `date_diff(a, b)` | safe arithmetic parser, never `eval` |
   | `save_note(title, text)`, `export(...)` | **write: needs approval** |

   The agent gets **no delete tool**.
5. **Tool gateway.** Validates arguments against pydantic schemas; resolves "the Workato worksheet" to a file id (ambiguous → clarify, none → not found); enforces `agent.max_steps` (~6) and a token budget per observation; drops repeated calls; runs every observation through `guard.find_injections` and wraps it as data.
6. **Recipes** (fixed tool sequences): list → catalog; summarize → card or map-reduce, each point citing section and page; compare → both cards + targeted reads + one compare call, reusing the contradiction check; extract → per-document extraction, answered as a table; generate → study notes, quiz or draft over what was read.
7. **Open ReAct loop.** Ollama's native `tools=` with qwen3, thinking on for these steps only (`agent.think_open`). Every call is traced. An invalid call gets one repair retry with the validation error, then the closest recipe takes over.
8. **Human-in-the-loop.** LangGraph `interrupt()` with a local SQLite checkpointer. Three cases: clarify which document was meant (the common one); confirm expensive jobs ("summarize all 12 files, ~3 min?"); approve any write. A rejection goes back to the reasoning step as feedback. `Engine.ask` becomes resumable: it yields an approval/clarify event and `Engine.resume(chat_id, decision)` continues the run.
9. **Respond.** Citations must point to text a tool actually read. The groundedness check runs on long synthesized answers. "Not found" stays for lookups and for documents that don't exist.
10. **Retrieval upgrades** (STATUS redesign ideas 1–2): grade each passage by its best score over the question's phrasings; translate the query into the document's language; split two-part questions; a second pass when the model refuses although graded passages exist.

Contract changes (fixed on `main` before agents start): `LLM.chat(messages, tools) -> AssistantTurn(content, tool_calls)` and a `tool` message role in `core/providers.py` (the fake LLM plays back scripted tool calls); `Task`, `Plan`, `ToolCall`, `ToolResult`, `DocCard` types; trace keys for the new nodes in `agent/types.py`; every switch in `core/config.py` + `config/default.yaml`.

Dependencies: `uv add langgraph-checkpoint-sqlite` (local SQLite checkpointer for human-in-the-loop). No other new packages; no new model downloads unless `qwen3:4b` is adopted.

### Tasks

- [x] **Decisions (user, 2026-10-12):** (1) summaries, comparisons, extractions and generated material cite the sections and pages they read; lookups keep the graded-passage rule and "not found" (in `CLAUDE.md`). (2) General knowledge: a clearly labeled local world-knowledge mode as a switch, off by default (`agent.world_knowledge`). (3) Document cards at ingest: on, as a switch (`ingest.enrich`).
- [x] **M0: contracts on `main`.** Provider tool-call protocol + fake, types, config switches, trace keys, checkpointer, `tasks.jsonl` format, stub signatures, agent prompts in `scripts/agents/prompts/` (and the new shared files and dependency in `_rules.md`). Done 2026-10-12: `LLM.chat` checked live with `qwen3:14b` (a correct native tool call in 0.3 s with thinking off, 2.1 s on); `CardStore` (migration 4) and `ChunkIndex.parents_for_file` implemented; stubs `ingest/enrich.py`, `agent/catalog.py`, `agent/tools.py`, `agent/gateway.py`; prompts `cards.md`, `tools.md`, `taskeval.md`.
- [ ] **M1: foundations, three agents in parallel.** (a) Document cards at ingest + card store; (b) tools + gateway; (c) `tasks.jsonl` + key-facts judge + `eval run --repeat N` + router labels.
- [ ] **M2: planner, recipes, transform.** Wired into the graph with the lookup path untouched.
- [ ] **M3: ReAct loop, human-in-the-loop, verify.**
- [ ] **M4: retrieval upgrades** (component 10), each measured on CUAD and personal.
- [ ] **M5: UI.** A live step list ("Reading Syllabus, pp. 1–12…"), approval and clarify cards, Summarize / Quiz me per file in the Library, the route and tool calls in the trace panel, section chips on summary answers, memory in Settings.
- [ ] **Eval:** ~60 task items, half in Hebrew: list 5, summarize 10, compare 8, extract 8, transform 8, multi-step 15, ambiguous (should clarify) 6. Metrics: router accuracy, task success (key-facts judge, 3 repeats), citation precision, valid tool-call rate, steps per task, latency per task type, injection items where a document tries to trigger a tool call. CUAD and personal lookup numbers must hold within noise.

| Milestone | Done when |
|---|---|
| M0 | Tests and mypy green with fakes |
| M1 | Each part has unit tests; cards built for the real vault |
| M2 | The four failing requests in STATUS (and their Hebrew versions) get correct, cited answers; lookups within noise |
| M3 | Multi-step set measured (success, steps, valid-call rate); approve / reject / clarify work end to end |
| M4 | Cross-lingual and two-part items improve; CUAD abstention holds |
| M5 | The hand walkthrough passes |

Latency targets: lookup ≤ 2 s (1.5 s today); list < 1 s; cached summary < 1 s; new summary first token < 10 s; compare < 15 s; open tasks < 30 s with steps shown live.

**Done when:** M0–M5 are done; the router, each task type and the open loop have measured numbers in RESULTS.md; lookup quality and latency hold.

Later (after Phase 6): a local vision model for scanned pages and figures (together with Phase 6 OCR), and long-term user memory.

## Phase 6: Hardening + privacy

Dependencies: `uv add --dev pytest-socket`

- [ ] **Network-blocked test:** run ingest + ask with all sockets blocked except `127.0.0.1`. It must pass.
- [ ] Telemetry audit: check every dependency for analytics and disable or remove any you find. Set `HF_HUB_OFFLINE=1` once the models are cached.
- [ ] Edge cases: scanned PDFs (no text layer → flag the file, don't fail silently), encrypted PDFs, empty or huge files, non-UTF-8 text, and files that change during indexing.
- [ ] Performance: batched embedding, and measure ingestion throughput (pages/sec).
- [ ] Resilience: if Ollama goes down mid-session, show a clear message and recover. Interrupted ingestion jobs resume after a restart.
- [ ] **Secret / PII scanner at ingest** (config: `ingest.secret_scan: off | flag | mask | exclude`): local regex patterns for API keys and tokens, private keys, passwords in config-like lines, card numbers (Luhn-checked), emails, phone and ID numbers. Matches are recorded per file in the registry and shown in the Library with a lock icon. The same pass flags prompt-injection patterns for Phase 4. No model, no network.
- [ ] Integration test pass on the full corpus.
- [ ] **Launch token:** a secret generated at each `vaultmind ui` start, required by the server (cookie set from the opening URL), so other programs on the machine can't use the app. The Host/Origin check (Phase 5) already stops other websites.
- [ ] **Index consistency:** re-indexing a file swaps its chunks atomically (today a question asked between the delete and the add can miss the file).
- [ ] Docs: run VaultMind in a separate browser profile without extensions; which browser features send page text out.

**Done when:** the network-blocked test passes in CI and none of the edge cases crashes the app.

## Phase 7: Packaging + showcase

- [ ] One-command launch for users (`uv tool install` or a launcher script); document the setup on Windows, macOS, and Linux.
- [ ] Optional: a packaged Windows build attached to GitHub Releases.
- [ ] README: demo GIF/video (30–60 s), final results table, architecture diagrams kept up to date.
- [ ] `docs/RESULTS.md` written up as a short report: what helped, what didn't, and why.
- [ ] CV bullets backed by the measured numbers (see below).

## Later: stretch ideas (only after Phase 7)

Not scheduled. Each one must keep the offline guarantee and pass the network-blocked test.

- [ ] **Encrypted vault:** passphrase-protected chat history (SQLCipher). Embeddings can be partly inverted back to text, so encrypting only the text columns is not enough; for the index, document OS disk encryption (BitLocker / FileVault / LUKS) rather than rolling our own.
- [ ] **Offline voice notes:** transcribe audio files in the vault locally (faster-whisper; model downloaded only in an explicit setup step) and index them with timestamps in place of pages.

Ruled out: **MCP** (as client or server). A client would let third-party tool servers handle document content, which the privacy guarantee can't cover; a server adds little before the corrective loop exists. Features like PDF export are built in directly instead.

---

## Risks

| Risk | Mitigation |
|---|---|
| torch on RTX 50 (Blackwell) requires CUDA 12.8+ builds | Install from the cu128+ index; fall back to CPU or an ONNX reranker if needed |
| PDF parsing quality (tables, multi-column, scans) | Include hard PDFs in the corpus from day 1; Docling as an optional parser |
| **PyMuPDF is AGPL-3.0** | Fine for an open-source repo. Before distributing a bundled binary, switch to `pypdfium2` (Apache/BSD) or Docling (MIT) behind the same parser interface |
| 14B model too slow for a responsive chat | Benchmark 8B vs. 14B in the eval; 8B for analyze/rewrite, 14B for generation |
| Indexed documents contain prompt-injection text | Passages are delimited as data, flagged at ingest, and measured with `injection` eval items (Phases 2, 4, 6) |
| An LLM judge for correctness is noisy | Spot-check by hand; report retrieval metrics, which are deterministic, as the headline numbers |
| A 14B local model makes unreliable tool calls | Lookups and common tasks never go through the open loop (fast paths + recipes); schema-validated arguments, one repair retry, then the closest recipe |
| Documents can now steer tool choice (injection) | Tools are read-only, writes always need approval, no delete tool, every tool observation goes through the injection guard; eval items where a document tries to trigger a tool call |
| Whole-document tasks don't fit in `num_ctx` | Document cards at ingest + map-reduce keep correctness independent of context size; a larger `num_ctx` (likely with Ollama's q8 KV cache) is only a speed lever |
| Scope creep | Out of scope: multi-user, auth, cloud sync, GraphRAG, multi-agent setups |

## CV bullet template

Fill in the numbers from `RESULTS.md`:

> Built **VaultMind**, a fully offline agentic RAG desktop app (Python, LangGraph, Ollama, LanceDB). Designed a corrective retrieval loop with hybrid search and cross-encoder grading that raised recall@5 from **X% → Y%** and abstention accuracy on unanswerable questions to **Z%**, and resisted **R%** of prompt-injection attacks planted in documents, measured with a custom evaluation harness, at **N s** median latency on a consumer GPU.
