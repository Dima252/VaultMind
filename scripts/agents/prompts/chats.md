# Task: Chat history store + export to Markdown/PDF (Phase 5, wave 1)

First read and follow `scripts/agents/prompts/_rules.md`. Then read `docs/STATUS.md` and the Phase 5 section of `docs/EXECUTION_PLAN.md`.

## Your files
- `src/vaultmind/store/chats.py`: implement the stub. Keep the public signatures and dataclasses; you may add private helpers.
- `src/vaultmind/app/export.py`: implement the stub.
- `tests/test_chats.py`, `tests/test_export.py` (new).

## Contract (read, don't edit)
- `store/registry.py` migration 3: the `chats` and `messages` tables. Use `messages.meta` (JSON text) for an assistant message's `sources`, `abstained`, `contradictions`, `unsupported`; user messages keep `{}`.
- `store/traces.py`: copy its connection setup (WAL, foreign keys, `migrate`, a lock, idempotent `close`). The ChatStore shares `vaultmind.sqlite` with `Registry` and `TraceStore`; `messages.run_id` references `runs(id)`.
- `agent/types.py` (`AgentResult`, `Contradiction`, `UnsupportedClaim`) and `agent/baseline.py` (`Source`, with `Path | None` file_path, `tuple` section_path and warnings, `cited_sources`).
- `app/engine.py`: `Engine.ask` calls `chats.history(chat_id)` before the run and `chats.add_exchange(chat_id, result, run_id)` after it, and yields the returned assistant `ChatMessage`.

## ChatStore
- Ids: uuid4 hex. Times: UTC ISO strings in the database, aware `datetime` in the dataclasses.
- `add_exchange`: seq continues from the last message; question = `result.question`, answer = `result.answer`; one transaction for both rows plus `updated_at` (and the auto-title, see the stub docstring). Serialize `Source` exactly enough to rebuild an equal `Source` (round-trip test, including `None` file path/page, Hebrew text, warnings).
- `history`: `[{"role": ..., "content": ...}]` for every message, oldest first.
- `delete` removes messages (cascade) but not runs.

## Export
- `answer_markdown`: a `## ` heading with the question, the answer text as is (keep `[n]` markers), then a `Sources` list with `[n] label` for the cited sources (`cited_sources(answer.content, answer.sources)`, or all sources if none are cited). For an abstained answer, list them as `Closest matches` instead. Add the contradictions and unsupported claims as short notes when present.
- `chat_markdown`: `# title`, an "Exported from VaultMind on <local date time>" line, then each exchange. Pair messages by order (user, assistant); a trailing unanswered user message is still exported.
- `markdown_pdf`: render offline with PyMuPDF (`pymupdf.Story` with HTML converted from the Markdown is a good fit; convert the small subset you produce yourself: headings, paragraphs, lists, bold. Don't add a Markdown library). It must not download anything. **Hebrew must come out legible and right-to-left**: set `dir="rtl"` on Hebrew paragraphs (detect by the first strong character) and use a font with Hebrew glyphs. Look for one on the system first (Windows: `C:/Windows/Fonts/arial.ttf`; macOS/Linux: common paths such as DejaVu or Noto), and fall back to PyMuPDF's built-in font for Latin-only text. Document what happens when no Hebrew-capable font is found. Return the PDF bytes.

## Tests (no models)
- ChatStore: create/get/recent ordering, rename (KeyError unknown), add_exchange (seq, auto-title only on first exchange and only for the default title, long/multiline questions), the Source/Contradiction/UnsupportedClaim round-trip, history, delete cascade keeps runs (save one with `TraceStore`), unknown chat → `[]`/`None`/KeyError as documented, and coexistence with `Registry` and `TraceStore` on the same file.
- Export: Markdown for answered/abstained/contradiction cases and a whole chat (hand-checked strings); `markdown_pdf` returns a valid PDF whose extracted text (`pymupdf.open(stream=...)`) contains the English text, and the Hebrew words when a Hebrew-capable font is present (skip that assertion otherwise, with a reason). Use synthetic text only.
