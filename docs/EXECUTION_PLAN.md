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
- [ ] Install Ollama, then `ollama pull qwen3:14b` and `ollama pull qwen3-embedding:0.6b`
- [ ] `uv run vaultmind doctor` passes
- [ ] Collect a **test corpus**: 15–30 real documents of mixed types (papers, manuals, a long report, slides exported to PDF, Markdown notes, at least one PDF with tables). The Phase 2 eval set is built from this corpus, so choose documents you know well.

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

> Status: code complete, 174 unit tests passing. Waiting on Ollama for the live end-to-end test (`uv run pytest -m ollama`) and a run over the real corpus.

## Phase 2: Evaluation harness

Goal: one command that scores any config. This turns "I think it's better" into numbers.

- [x] Dataset format `eval/datasets/<name>.jsonl`: `{question, answer, source_file, source_page, answerable}`
- [ ] `vaultmind eval generate`: the LLM drafts questions from random chunks. **Verify each one by hand.** Aim for 40–60 questions, about 20% of them unanswerable (on-topic, but the answer isn't in the corpus).
- [ ] Include tough cases: exact-term lookups (codes, names), multi-hop questions, vague phrasing, and follow-up questions that depend on chat history.
- [~] Metrics: recall@k and MRR (retrieval), answer correctness against the reference (LLM judge, spot-checked by hand), abstention accuracy, latency p50/p95. *(Done: recall@k, MRR, abstention, percentiles in `eval/metrics.py`. To do: LLM-judge correctness.)*
- [ ] `vaultmind eval --config a.yaml --config b.yaml` runs each config and prints a comparison table, saving raw results to `eval/results/`.
- [ ] Record the **baseline** in `docs/RESULTS.md`.

**Done when:** the baseline numbers are recorded, and re-running the eval gives the same retrieval numbers.

## Phase 3: Retrieval quality

Goal: raise recall@5. Measure each step separately with an ablation config.

Dependencies: `uv add sentence-transformers`, plus torch from the CUDA 12.8+ index (the RTX 50 series is Blackwell and needs cu128 or newer).

- [ ] **Hybrid search:** LanceDB full-text index on child text, run alongside vector search, fused with reciprocal rank fusion. Config: `retrieval.hybrid`.
- [ ] **Reranker:** `core/rerank.py` wrapping a sentence-transformers `CrossEncoder` (bge-reranker-v2-m3) on the GPU. Rerank `candidates` (20) down to `top_k` (5). Config: `retrieval.rerank`.
- [ ] **Small-to-big:** search over children, deduplicate by parent, and send parent sections to the LLM. Compare answer correctness, not just recall.
- [ ] Tune `rerank_threshold` against the eval set: plot the threshold against answerable recall and unanswerable abstention, then pick the best balance.
- [ ] Record each ablation in `RESULTS.md`.

**Done when:** each technique has a measured contribution, whether positive or not. Keep only what helps.

## Phase 4: Agentic corrective loop

Goal: replace the linear baseline with the LangGraph state machine.

Dependencies: `uv add langgraph`

- [ ] Graph state: question, history, queries tried, candidates, graded passages, rewrite count, trace events.
- [ ] **Analyze node** (one structured call): standalone query (resolves chat follow-ups), 2–3 query variants, `needs_retrieval`.
- [ ] **Retrieve node:** hybrid retrieval for each query variant, fused.
- [ ] **Grade node:** reranker scores against the threshold. No per-chunk LLM calls.
- [ ] **Rewrite node:** given the failed queries and what was found, produce a different query. Stop after `agent.max_rewrites`.
- [ ] **Generate node:** streamed, cites only graded passages.
- [ ] **Not-found node:** an honest "not in your documents" answer plus the closest matches.
- [ ] Optional **groundedness check** node (config flag, off by default): checks each claim against its cited passage.
- [ ] **Traces:** every node emits events (inputs, outputs, scores, duration), saved to a SQLite `traces` table. `vaultmind ask --trace` prints them.
- [ ] Eval: loop on vs. off. Report how many failed queries the rewrite recovered, what it costs in latency, and how abstention changes.

**Done when:** the loop beats or matches the Phase 3 config on correctness and abstention, and its latency cost is measured.

## Phase 5: Desktop UI + live ingestion

Goal: a polished app that calls the engine directly, with no API layer.

Dependencies: `uv add "nicegui[native]" watchdog`

- [ ] `vaultmind ui` opens a native window (`ui.run(native=True)`).
- [ ] **Chat:** streamed answers; citation chips that open the source PDF at the right page; conversation history saved in SQLite.
- [ ] **Trace panel:** a collapsible timeline under each answer showing queries, passage scores, rewrites, and per-step timing.
- [ ] **Library:** drag-and-drop upload; per-file status (queued / indexing / ready / failed with reason); re-index and remove actions.
- [ ] **Live folder watching:** a watchdog observer feeds the ingestion job queue in the background, so the UI never blocks.
- [ ] **Settings:** watched folders, models, and retrieval/agent switches (written to `config/local.yaml`).
- [ ] Optional: an **Eval** page that runs the test set and shows the comparison table in-app.
- [ ] Load models at startup so the first question isn't slow.

**Done when:** a new user can drop files in, watch them index, ask a question, and inspect the trace without touching the CLI.

## Phase 6: Hardening + privacy

Dependencies: `uv add --dev pytest-socket`

- [ ] **Network-blocked test:** run ingest + ask with all sockets blocked except `127.0.0.1`. It must pass.
- [ ] Telemetry audit: check every dependency for analytics and disable or remove any you find. Set `HF_HUB_OFFLINE=1` once the models are cached.
- [ ] Edge cases: scanned PDFs (no text layer → flag the file, don't fail silently), encrypted PDFs, empty or huge files, non-UTF-8 text, and files that change during indexing.
- [ ] Performance: batched embedding, and measure ingestion throughput (pages/sec).
- [ ] Resilience: if Ollama goes down mid-session, show a clear message and recover. Interrupted ingestion jobs resume after a restart.
- [ ] Integration test pass on the full corpus.

**Done when:** the network-blocked test passes in CI and none of the edge cases crashes the app.

## Phase 7: Packaging + showcase

- [ ] One-command launch for users (`uv tool install` or a launcher script); document the setup on Windows, macOS, and Linux.
- [ ] Optional: a packaged Windows build attached to GitHub Releases.
- [ ] README: demo GIF/video (30–60 s), final results table, architecture diagrams kept up to date.
- [ ] `docs/RESULTS.md` written up as a short report: what helped, what didn't, and why.
- [ ] CV bullets backed by the measured numbers (see below).

---

## Risks

| Risk | Mitigation |
|---|---|
| torch on RTX 50 (Blackwell) requires CUDA 12.8+ builds | Install from the cu128+ index; fall back to CPU or an ONNX reranker if needed |
| PDF parsing quality (tables, multi-column, scans) | Include hard PDFs in the corpus from day 1; Docling as an optional parser |
| **PyMuPDF is AGPL-3.0** | Fine for an open-source repo. Before distributing a bundled binary, switch to `pypdfium2` (Apache/BSD) or Docling (MIT) behind the same parser interface |
| 14B model too slow for a responsive chat | Benchmark 8B vs. 14B in the eval; 8B for analyze/rewrite, 14B for generation |
| An LLM judge for correctness is noisy | Spot-check by hand; report retrieval metrics, which are deterministic, as the headline numbers |
| Scope creep | Out of scope: multi-user, auth, cloud sync, GraphRAG, multi-agent setups |

## CV bullet template

Fill in the numbers from `RESULTS.md`:

> Built **VaultMind**, a fully offline agentic RAG desktop app (Python, LangGraph, Ollama, LanceDB). Designed a corrective retrieval loop with hybrid search and cross-encoder grading that raised recall@5 from **X% → Y%** and abstention accuracy on unanswerable questions to **Z%**, measured with a custom evaluation harness, at **N s** median latency on a consumer GPU.
