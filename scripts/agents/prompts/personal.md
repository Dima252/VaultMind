# Task: The user's own documents: parsing check + eval questions (Phase 0/2 leftovers, for Phase 4)

First read and follow `scripts/agents/prompts/_rules.md`. Then read `docs/STATUS.md` and the dataset format in `src/vaultmind/eval/dataset.py` (note the new `history` field).

## Privacy (read twice)
The documents in `C:/Users/dimat/Documents/GitHub/VaultMind/vault/personal/` belong to the user: course material in Hebrew (algorithms lectures, an OS summary, assignments) and some English Workato PDFs. The repo is **public**.
- Never copy their text into code, tests, commit messages or docs. Refer to files generically ("the 69 MB Hebrew OS summary"), not by quoting content.
- Parser tests use **synthetic** fixtures you generate in the test (e.g. a small Hebrew PDF made with `pymupdf`), never the user's files.
- The question file is written **only** to the main checkout: `C:/Users/dimat/Documents/GitHub/VaultMind/eval/datasets/personal.jsonl`. It's gitignored there. Never create it in your worktree or commit it.

## Your files
- `src/vaultmind/ingest/parsers.py`: fixes only, keep the signatures.
- `tests/test_parsers.py`.
- Outside git: `C:/Users/dimat/Documents/GitHub/VaultMind/eval/datasets/personal.jsonl`.

## Setup
1. `uv sync --extra rerank` in your worktree (needed for real retrieval). Ollama runs locally with `qwen3:14b` and `qwen3-embedding:0.6b`. The reranker is already in the Hugging Face cache.
2. Create `config/local.yaml` in your worktree (gitignored). Set `paths.vault_dir` to the main checkout's `vault` folder (absolute path, so `source_file` values look like `personal/<name>.pdf`) and `paths.data_dir: ./data-personal`.
3. `uv run vaultmind ingest C:/Users/dimat/Documents/GitHub/VaultMind/vault/personal`.

## Part 1: parsing check
For every file, record: pages, parse result (ok / `ParseError` reason), chunk count, and whether the text reads correctly. Check this yourself by inspecting chunks (`ChunkIndex` in `./data-personal/index`).
- **Hebrew direction:** right-to-left PDFs often extract in visual order (reversed words or letters, mirrored parentheses). If the text comes out reversed, fix it in `parsers.py`, e.g. with a PyMuPDF extraction flag or by detecting and reordering RTL lines. Keep English and mixed Hebrew/English lines correct (code, formulas and numbers inside Hebrew text). Add synthetic tests.
- Slides (lecture PDFs): check that headings and bullets survive; a heading on every slide is fine.
- Tables (the Workato formulas PDF): check that table rows stay together and readable.
- The 69 MB OS summary: if it's mostly scanned images it should raise `ParseError` ("no text layer"), not hang or produce junk. Measure how long it takes. If it's slow but valid, report the time.
- After fixes, re-ingest (delete `./data-personal` first) and re-check.

Report a table of the files (generic descriptions + numbers) in your summary and commit body.

## Part 2: questions (~25), for the user to verify
Write `personal.jsonl` (one `EvalItem` per line; build items with the `EvalItem` model and write them with `save_dataset` so they validate). Mix:
- ~8 **exact lookups**: a fact, number, definition, date or deadline. Tags `exact-term` plus `hebrew` or `english`.
- ~4 **tables**: a value from a table row. Tag `table`.
- ~4 **multi-hop**: needs two documents or two distant sections, e.g. comparing two lectures or two Workato documents. Tag `multi-hop`. `source_file` = the main one; put spans from both in `evidence`.
- ~4 **follow-ups**: `history` has 1–2 earlier turns (user question + a short correct assistant answer), and the question only makes sense with them ("and what is its running time?"). Tag `follow-up`.
- ~2 **cross-lingual**: an English question about Hebrew content (or the reverse). Tag `cross-lingual`.
- ~4 **unanswerable**: plausible questions on the same topics whose answer is in none of the files. Search the chunks to confirm it's absent. `answerable: false`, tag `unanswerable`.

Rules:
- Questions in the language a user would ask them (mostly Hebrew for the Hebrew documents).
- `answer`: short and correct.
- `evidence`: spans copied **verbatim from the indexed chunk text** (so evidence recall can match them), ideally 1–2 sentences each.
- `source_page` when known.

Then run `uv run vaultmind eval run -d C:/Users/dimat/Documents/GitHub/VaultMind/eval/datasets/personal.jsonl -c <a config with agent.loop: false>` from your worktree as a smoke test, and report the metrics in your summary. The loop isn't implemented in your worktree, so use the linear pipeline.

Your summary must say clearly that **the user has to verify the answers** in `personal.jsonl`, and list the 5 questions you're least sure about.
