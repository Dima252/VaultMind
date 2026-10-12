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

- [ ] `vaultmind ui` opens a native window (`ui.run(native=True)`).
- [ ] **Chat:** streamed answers; citation chips that open the source PDF at the right page; conversation history saved in SQLite.
- [ ] **Trace panel:** a collapsible timeline under each answer showing queries, passage scores, rewrites, and per-step timing.
- [ ] **Library:** drag-and-drop upload; per-file status (queued / indexing / ready / failed with reason); re-index and remove actions.
- [ ] **Live folder watching:** a watchdog observer feeds the ingestion job queue in the background, so the UI never blocks.
- [ ] **Settings:** watched folders, models, and retrieval/agent switches (written to `config/local.yaml`).
- [ ] Optional: an **Eval** page that runs the test set and shows the comparison table in-app.
- [ ] Load models at startup so the first question isn't slow.
- [ ] **Highlighted citations:** clicking a citation opens the PDF at the page with the supporting text highlighted (PyMuPDF `search_for` on the cited child's text + a highlight annotation on a temp copy; never modify the user's file). Markdown/text sources scroll to and highlight the passage.
- [ ] **Export:** save an answer or a whole chat as Markdown or PDF, with its citations listed as file + page.
- [ ] **Privacy badge:** a status indicator that checks the app's own connections (`psutil.Process().net_connections()`) every few seconds and shows "localhost only". It turns red and names the remote address if any non-loopback connection appears.

**Done when:** a new user can drop files in, watch them index, ask a question, and inspect the trace without touching the CLI.

## Phase 6: Hardening + privacy

Dependencies: `uv add --dev pytest-socket`

- [ ] **Network-blocked test:** run ingest + ask with all sockets blocked except `127.0.0.1`. It must pass.
- [ ] Telemetry audit: check every dependency for analytics and disable or remove any you find. Set `HF_HUB_OFFLINE=1` once the models are cached.
- [ ] Edge cases: scanned PDFs (no text layer → flag the file, don't fail silently), encrypted PDFs, empty or huge files, non-UTF-8 text, and files that change during indexing.
- [ ] Performance: batched embedding, and measure ingestion throughput (pages/sec).
- [ ] Resilience: if Ollama goes down mid-session, show a clear message and recover. Interrupted ingestion jobs resume after a restart.
- [ ] **Secret / PII scanner at ingest** (config: `ingest.secret_scan: off | flag | mask | exclude`): local regex patterns for API keys and tokens, private keys, passwords in config-like lines, card numbers (Luhn-checked), emails, phone and ID numbers. Matches are recorded per file in the registry and shown in the Library with a lock icon. The same pass flags prompt-injection patterns for Phase 4. No model, no network.
- [ ] Integration test pass on the full corpus.

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
| Scope creep | Out of scope: multi-user, auth, cloud sync, GraphRAG, multi-agent setups |

## CV bullet template

Fill in the numbers from `RESULTS.md`:

> Built **VaultMind**, a fully offline agentic RAG desktop app (Python, LangGraph, Ollama, LanceDB). Designed a corrective retrieval loop with hybrid search and cross-encoder grading that raised recall@5 from **X% → Y%** and abstention accuracy on unanswerable questions to **Z%**, and resisted **R%** of prompt-injection attacks planted in documents, measured with a custom evaluation harness, at **N s** median latency on a consumer GPU.
