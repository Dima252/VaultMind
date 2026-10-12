# Task: Highlighted citations, privacy check, settings persistence (Phase 5, wave 1)

First read and follow `scripts/agents/prompts/_rules.md`. Then read `docs/STATUS.md` and the Phase 5 section of `docs/EXECUTION_PLAN.md`.

## Your files
- `src/vaultmind/app/highlight.py`, `src/vaultmind/app/privacy.py`, `src/vaultmind/app/settings.py`: implement the stubs, keep the signatures.
- `tests/test_highlight.py`, `tests/test_privacy.py`, `tests/test_settings.py` (new).

## Contract (read, don't edit)
- `agent/baseline.py`: `Source` has `file_path`, `page` (1-based, PDFs only), `text` (the cited child chunk's text as the parser produced it: Markdown from `pymupdf4llm`, possibly with `**`, `#`, table pipes, and for Hebrew pages text rebuilt into logical order by `ingest/parsers.py`), `chunk_id`.
- `core/config.py`: `Config`, `LOCAL_CONFIG`, `load_config`. A test asserts `config/default.yaml` equals `Config()`.

## Highlight (PDF + text)
- `render_pdf_citation`: open the PDF read-only, find the passage on `page` and add highlight annotations **in memory** (never save over the user's file), render the page to PNG at `zoom`, return `PageImage`. Out-of-range page → clamp to a valid page. Encrypted/unreadable → raise `ValueError` with a readable message.
- Finding the passage is the hard part. `page.search_for` needs text as it appears on the page, and the chunk is Markdown that may span a page break. Strategy to implement and test:
  1. Strip Markdown markup (`**`, `__`, `#` headings, `|` table separators, list markers, `<br>`), normalize whitespace.
  2. Split into sentences/lines; search each piece of reasonable length (say 4+ words, cut long pieces into windows of ~8 words) on the page; highlight every hit. `highlighted` is True when at least one piece was found.
  3. Hebrew: the page's stored text order may differ from the parser's logical order. Try each piece as is, and if nothing matches, try word-level matching (search individual words of 3+ letters and highlight the words that occur in a contiguous region; keep it simple and bounded). If it still fails, return the page unhighlighted (`highlighted=False`); never raise for "not found".
  4. Also try `page - 1` and `page + 1` when nothing is found on `page` (chunks can straddle pages) and show the page where the passage was found.
- `highlighted_pdf_copy`: same annotations, saved as a new file in `out_dir` (name it after the original + `-highlight.pdf`), returns its path.
- `find_passage`: map the passage into the original document text, tolerant of whitespace and Markdown markup differences (build a normalized copy of the document with an index map back to original offsets). Return the original-offset span or None.
- Tests: generate PDFs in `tmp_path` with PyMuPDF (English text across two pages, a passage with Markdown markup, a passage that doesn't exist, a clamped page number); check that `highlighted` is True/False as expected, the PNG decodes (starts with the PNG signature), and that the original file is byte-for-byte unchanged afterwards. For Hebrew, insert synthetic Hebrew text with a Hebrew-capable system font if available (skip with a reason otherwise) and check that it doesn't raise. `find_passage` cases: exact, whitespace differences, markup differences, not found.

## Privacy
- `check(pid)`: `psutil.Process(pid)` plus `.children(recursive=True)`; for each, `net_connections(kind="inet")` (older psutil: `connections`). Ignore connections without `raddr` (listening) and loopback remotes. For each remaining one, record `"<ip>:<port> (<process name>)"`, deduplicated and sorted. Processes that vanish or deny access (`psutil.NoSuchProcess`, `AccessDenied`) are skipped, not errors. `local_only = not remote`.
- `is_loopback`: use `ipaddress`; handle IPv4-mapped IPv6 and scoped addresses (`fe80::1%3` is not loopback, but don't crash on it); anything unparsable → False.
- Tests: `is_loopback` table; `check` with monkeypatched `psutil.Process` returning fake connections (loopback, remote, listening, a child that raises `NoSuchProcess`); and one real call on the test process that only asserts it returns a `PrivacyStatus`.

## Settings
- `overrides`: compare `cfg.model_dump(mode="json")` with `Config().model_dump(mode="json")` recursively; include only differing leaves, keep section nesting.
- `save_local`: validate by round-tripping through `Config.model_validate`, write YAML (`sort_keys=False`, `allow_unicode=True`) with a header comment saying the file is written by the app's Settings page and holds only overrides, atomically (temp file in the same folder + `os.replace`). Create the parent folder if needed. An empty override set writes the header plus `{}`, so `load_config` still works (check this).
- `needs_restart` / `needs_reindex` per the docstrings. Dotted keys like `models.llm`.
- Tests: overrides empty for defaults; nested changes; Path values serialize as strings and load back equal; save → `load_config(path)` equals the cfg; atomic write leaves no temp files; restart/reindex classification.
