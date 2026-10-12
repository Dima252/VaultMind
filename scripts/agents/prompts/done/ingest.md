# Task: Parsing and chunking

First read and follow `scripts/agents/prompts/_rules.md`.

## Your files
- `src/vaultmind/ingest/parsers.py` (implement the stub)
- `src/vaultmind/ingest/chunker.py` (implement the stub)
- `tests/test_parsers.py`, `tests/test_chunker.py` (new)

## Goal
Turn files into `ParsedDocument`s, then into `ParentSection`s and `ChildChunk`s (types in `src/vaultmind/core/models.py`). Retrieval searches small child chunks; the LLM later reads the larger parent section ("small-to-big"). Citations need correct page numbers and heading paths.

## Parsers (`parse(path)`)
- Dispatch on lowercase extension; anything outside `SUPPORTED_EXTENSIONS` raises `UnsupportedFileError`.
- **PDF**: `pymupdf4llm.to_markdown(..., page_chunks=True)` gives per-page Markdown and metadata. Check the installed version's API in `.venv`.
  - Track the heading hierarchy (`#`, `##`, ...) **across pages**, so a section that continues onto the next page keeps its path.
  - Emit one `Block` per heading-delimited span per page, with a 1-based `page`.
  - Encrypted/password-protected PDFs and corrupt files raise `ParseError`. So does a PDF with no extractable text (likely scanned), with a message saying OCR isn't supported yet.
- **Markdown**: headings from ATX `#` lines. Ignore `#` inside fenced code blocks. `page=None`.
- **Text**: blocks split on blank-line paragraph groups, `section_path=()`, `page=None`.
- Decode text as UTF-8, handling a BOM. Undecodable bytes raise `ParseError` with a clear message (better encoding detection comes in Phase 6).
- `title`: first H1 → PDF metadata title (if non-empty) → file stem.
- Drop empty/whitespace-only blocks. Never return a document whose blocks are all empty; raise `ParseError` instead.

## Chunker (`chunk_document(doc, file_id, cfg)`)
- `estimate_tokens(text)`: `ceil(len(text) / 4)`. It's deliberately simple and needs no tokenizer download.
- **Parents**: accumulate consecutive blocks up to `cfg.parent_tokens`. Prefer starting a new parent at a heading change once the current parent is at least half full. A single block bigger than `parent_tokens` is split at paragraph, then sentence, then hard boundaries. `section_path` = path of the parent's first block; `page_start`/`page_end` cover its blocks.
- **Children**: split each parent into windows of about `cfg.child_tokens` with `cfg.overlap` fractional overlap. Break at paragraph or sentence boundaries where possible. Child `page` = page where the child's text starts. `section_path` = section of the block it starts in.
- `embed_text = " > ".join((doc.title, *section_path)) + "\n\n" + text`
- Ids: parent `f"{file_id}:p{i}"`, child `f"{parent_id}:c{j}"`. Output must be **deterministic** (same input → identical output).
- Empty document → `([], [])`.

## Tests
- Build PDFs in `tmp_path` with `pymupdf` (`import pymupdf`). Use larger font sizes for headings so pymupdf4llm detects them. Cover: multi-page with a heading continuing across pages, an encrypted PDF, a PDF with no text, Markdown with a code fence containing `#`, text paragraphs, BOM, invalid UTF-8, unsupported extension.
- Chunker properties: no child exceeds `child_tokens` by more than 10%, and no parent exceeds `parent_tokens` by more than 10%; every parent's text is covered by its children; ids are unique; output is deterministic; pages and section paths are correct for a multi-section, multi-page document; overlap is present between consecutive children.
