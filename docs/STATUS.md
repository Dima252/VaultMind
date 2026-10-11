# VaultMind: Status and Handoff

_Last updated: 2026-10-12._

This is the starting point for anyone (human or agent) continuing the work. Read it after `CLAUDE.md`. The full roadmap with checkboxes is in [EXECUTION_PLAN.md](EXECUTION_PLAN.md).

## Where things stand

| Phase | State |
|---|---|
| 0. Project setup | ✅ Done. Eval corpus: 25 CUAD contracts + 10 adversarial documents. Your own mixed-type documents still optional |
| 1. Ingestion + baseline RAG | ✅ Done and verified against real models |
| 2. Evaluation harness | ✅ Done. Baseline in [RESULTS.md](RESULTS.md), reproducible. `eval generate` deferred |
| 3–7 | ⬜ Not started |

**Next task:** Phase 3, retrieval quality. The baseline's main CUAD failure is that the right clause isn't retrieved (evidence recall@5 0.708; 11 of 14 wrong abstentions had no gold clause in context), which is what hybrid search + reranking target. Re-run `vaultmind eval run -d cuad -d adversarial` after each change and add a column to `RESULTS.md`.

**Eval corpus (2026-10-12):** CUAD's `data.zip` is copied to `eval/corpora/cuad/` (gitignored, 18 MB, from the `../cuad` repo). `uv run vaultmind eval import-cuad` writes 25 contracts to `vault/cuad/` and `eval/datasets/cuad.jsonl`; the same seed always gives the same subset. Ingest takes about 20 s. All 12 unanswerable items were checked by hand against the contract text (two bad labels found and fixed via `RELATED` categories in `eval/cuad.py`). The full CUAD release with the 510 original PDFs is at `eval/corpora/cuad/CUAD_v1.zip` (md5 verified) for the optional txt-vs-PDF comparison. CUAD is CC BY 4.0: credit it (Hendrycks et al., NeurIPS 2021, The Atticus Project) in `RESULTS.md` and the README.

## What works today

```powershell
uv run vaultmind doctor                 # Ollama reachable + both models pulled
uv run vaultmind ingest [PATH]          # index vault/ (or PATH); incremental
uv run vaultmind ask "question"         # streamed answer with [n] citations
uv run vaultmind config                 # effective config
uv run vaultmind eval import-cuad       # CUAD subset -> vault/cuad/ + eval/datasets/cuad.jsonl
uv run vaultmind eval install-adversarial  # poisoned + contradicting docs -> vault/adversarial/
uv run vaultmind eval run -d cuad -d adversarial [-c a.yaml -c b.yaml] [--no-judge] [--limit N]
```

Verified on the dev machine (Windows 11, RTX 5070 Ti 16 GB, Ollama with `qwen3:14b` + `qwen3-embedding:0.6b`):

- 261 unit tests pass (fakes, no GPU). 6 live tests pass with `uv run pytest -m ollama` (about 75 s including model load).
- Ingesting 3 Markdown docs → 27 chunks in about 2 s. Re-running skips unchanged files.
- `ask` answers in about 2 s with a warm model and `think: false`. It cites the correct file and section, and answers "I couldn't find this in your documents." for out-of-corpus questions, listing the closest matches.
- ruff, ruff format, and mypy `--strict` are clean. CI (`.github/workflows/ci.yml`) runs the same checks plus unit tests.

## What was built (module map)

| Module | What it does | Notes |
|---|---|---|
| `core/config.py` + `config/default.yaml` | Typed config (pydantic, `extra="forbid"`). Resolution: `--config` → `$VAULTMIND_CONFIG` → `config/local.yaml` → `config/default.yaml` | A test asserts the YAML equals the built-in defaults; change both together |
| `core/models.py` | Shared dataclasses: `Block`, `ParsedDocument`, `ParentSection`, `ChildChunk`, `SearchHit`, `FileRecord`, `file_id_for()` | The contracts between modules |
| `core/providers.py` | `LLM`, `Embedder`, `Reranker` protocols | Pipeline code never imports `ollama` directly |
| `core/ollama.py` | `OllamaLLM`, `OllamaEmbedder`, `OllamaUnavailableError` | Always sends `num_ctx` + `keep_alive`. Schema calls use `format=`, `temperature 0`, `think=False`. Strips `<think>` text, even when split across stream deltas. Batches embeddings (32) |
| `ingest/parsers.py` | PDF (pymupdf4llm, `page_chunks=True`, no OCR), Markdown, text → `ParsedDocument` | Heading path carries across PDF pages. Scanned, encrypted and non-UTF-8 files raise `ParseError` |
| `ingest/chunker.py` | Small-to-big chunking: parents ≈1200 tokens, children ≈256 tokens, 15% overlap. Tokens = `ceil(len/4)` | Children are exact slices of parent text. `embed_text` = `Title > Section\n\n` + text, without repeating the title |
| `ingest/pipeline.py` | `ingest_path()`: sha256 skip, parse → chunk → embed → `delete_file` + `add`, removes deleted files | Parse failures are recorded and **retried** next run. Other errors (Ollama down) mark the file failed and stop the run |
| `store/index.py` | LanceDB tables `children` (vectors), `parents`, `meta` | Cosine similarity score. Refuses a different embedder (`EmbedderMismatchError`). Upserts by id. No telemetry (checked) |
| `store/registry.py` | SQLite `files` table, WAL, thread-safe, `PRAGMA user_version` migrations | |
| `agent/baseline.py` | Linear RAG: embed → `vector_search(top_k)` → dedupe to parents → numbered prompt → answer | `NOT_FOUND` string, `is_abstention()`, `cited_sources()`. Citations use the best child's page and section |
| `eval/dataset.py` | `EvalItem` + JSONL load/save with line-numbered errors | Validator: unanswerable ⇒ no source; answerable ⇒ `source_file` required |
| `eval/metrics.py` | `recall_at_k`, `mrr`, `abstention_accuracy`, `percentile`, `source_matches`, `evidence_recall_at_k`, `evidence_mrr`, `injection_resistance`, `correctness_score` | Docstrings state exact formulas (cite them in the README) |
| `eval/judge.py` | LLM judge: one schema call, verdict correct / partial / incorrect + reason, one retry, `JudgeError` | Prompt in module constants. Ignores answer content the reference doesn't cover. 11 of 15 hand-checked verdicts agreed |
| `eval/runner.py` | `run_config` (timed retrieve + generate + judge), `compute_metrics`, `check_compatible`, `save_report`, `format_table` | Currently runs the baseline pipeline; Phase 4 will need a pipeline switch |
| `eval/adversarial.py` + `eval/fixtures/adversarial/` | 4 poisoned contracts (override, fake system block, exfiltration, subtle) + 3 contradiction pairs; `install_adversarial()` | Fictional names, `.invalid` domains. Payload codes like `ZORBLAX-4417` must never appear in answers |
| `eval/cuad.py` | CUAD importer: seeded contract subset -> `vault/cuad/*.txt`, questions that name their contract, spans as `evidence`, absent clauses as unanswerable | Owns `vault/cuad/`: stale `.txt` files there are removed on re-import |
| `cli.py` | `config`, `doctor`, `ingest`, `ask`, `eval import-cuad`, `eval install-adversarial`, `eval run` | Forces `errors="replace"` on stdout (Windows cp1255 console) |

Storage lives in `paths.data_dir` (default `./data`): `index/` (LanceDB) and `vaultmind.sqlite`. Both are gitignored.

## Decisions already made (don't re-litigate without measurements)

- **No API layer.** The UI (NiceGUI, `native=True`) will call the engine in-process. The engine never imports `ui/`.
- **LanceDB** over Chroma (built-in full-text search for hybrid retrieval, no telemetry). **SQLite** for app state.
- **Reranker scores grade chunks**, not per-chunk LLM calls (Phase 3/4).
- **No `tiktoken`**: it downloads its tokenizer at runtime, which breaks the offline rule.
- **`think: false` by default.** Qwen3 thinking adds latency. It's a config switch, so the eval can measure its effect.
- **Python 3.12** (3.14 wheels for torch/lancedb aren't reliable).
- **The name is VaultMind** (the original brief called it H.I.M.). **MIT license.** The repo is **public**.
- **No MCP**, as client or server (decided 2026-10-11). Third-party tool servers would handle document content outside the privacy guarantee. Output features such as PDF export are built in directly.
- **Security as a feature** (added 2026-10-11): prompt-injection hardening with a measured resistance rate, a live "localhost only" privacy badge, a secret/PII scanner at ingest, highlighted citations, and an optional contradiction check. Each is a task in its phase in the plan.

## What's left

### Phase 0
- [x] Main corpus: 25 CUAD contracts via `uv run vaultmind eval import-cuad` (see below).
- [x] Adversarial documents: 4 poisoned contracts and 3 contradicting pairs (`eval install-adversarial`).
- [ ] Optional: 5–10 of your own mixed-type documents (PDF with tables, slides, Markdown) for parser coverage.

### Phase 2: Evaluation harness (done 2026-10-12)
All core items are done; see the plan's checkboxes and [RESULTS.md](RESULTS.md). Left over, none blocking Phase 3:
- [ ] `vaultmind eval generate`: LLM-drafted questions over your own documents, for human review.
- [ ] Hand-written multi-hop and chat follow-up questions (needed before Phase 4's history handling).
- [ ] Optional: `import-cuad --format pdf` using `CUAD_v1.zip`, recording page numbers, to measure the txt vs PDF parsing gap.
- [ ] Judge reliability: if correctness noise matters for a decision, switch the judge to a key-facts checklist (list the reference's facts, mark each covered or contradicted, compute the verdict in code).

### Phases 3–7
See [EXECUTION_PLAN.md](EXECUTION_PLAN.md): hybrid search + reranker + small-to-big ablations (3), the LangGraph corrective loop + traces + injection guard + contradiction check (4), the NiceGUI desktop app + folder watching + highlighted citations + export + privacy badge (5), hardening + network-blocked privacy test + secret/PII scanner (6), packaging + demo + write-up (7). Stretch ideas (encrypted vault, voice notes) are listed after Phase 7.

## Known limitations (from the agents' reports and testing)

- **Parsing:** code fences are tracked per PDF page; setext (`===`/`---`) headings aren't recognized; sentence splitting doesn't know abbreviations; PDF headings are only detected when their font is larger. UTF-8 only, no OCR (Phase 6).
- **Index:** exact vector scan with no ANN index (fine at this scale). `add` writes parents and children in two steps, not one transaction.
- **Phase 3 warning:** LanceDB full-text tokenizers that load language dictionaries may **download** them. Use the default, simple or ngram tokenizer.
- **Eval:** ranks aren't deduplicated before the cut at k. Every JSONL line must include `source_file` and `source_page` keys (they can be `null`).
- **Baseline:** abstention is detected by a prefix match on `NOT_FOUND`. If the model rephrases it, it counts as an answer: 2 of 60 CUAD answers in the baseline run. A file whose H1 equals its file name gets a label like "CLAUDE.md, CLAUDE.md".
- Embedding batch size (32) isn't in the config yet.
- `vault/README.md` gets indexed like any document, so it shows up as a (harmless) distractor. Consider skipping it in ingest.
- CUAD contract names come from EDGAR filer names (`Aspirityholdingsllc Outsourcing Agreement (2012)`), which often differ from the party names inside the contract. Questions find the right contract mostly through the title prefix on every chunk. That's realistic (users refer to files by name), but keep it in mind when reading results.

## Environment gotchas (Windows dev machine)

- **Console code page is cp1255.** Keep CLI output ASCII (`[ OK ]`/`[FAIL]`, not ✓/✗). `cli.py` sets `errors="replace"` as a safety net.
- **`ollama` may not be on PATH** in a new shell. The executable is `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`.
- **Line endings:** `core.autocrlf=true`, so the "LF will be replaced by CRLF" warnings are harmless.
- **RTX 50 series (Blackwell):** Phase 3 needs torch from the CUDA 12.8+ index (`cu128`) for the reranker.
- Live tests (`-m ollama`) are excluded from default `pytest` runs and from CI.

## How the work was split (for reuse)

Phase 1 was built by four parallel agents, each in its own git worktree. The interfaces were fixed in stubs first so the merges were conflict-free; an integration pass then wired everything together.

- Task prompts: `scripts/agents/prompts/` (`_rules.md` holds the shared rules).
- In this session the agents were run with Claude Code's built-in subagents (`isolation: worktree`). The PowerShell scripts in `scripts/agents/` (`setup.ps1` → `start.ps1` → `merge.ps1`) do the same thing by hand. They pass a syntax check but **have not been run end to end yet**.
- To reuse for a later phase: write new stubs and contracts first, add one prompt per agent, and update `$Agents` in `scripts/agents/_common.ps1`.
- Phase 2 used three agents (runner, judge, adversarial); the shared metrics and CLI were done on `main` before and after. **Gotcha:** Claude Code's worktree agents branched from the commit the session started on, not current `main`, so each had to `git merge --ff-only main` to see the contracts. Put that step in the launch prompt.
