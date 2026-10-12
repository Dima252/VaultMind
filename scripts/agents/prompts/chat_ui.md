# Task: Chat page, trace view, citation viewer, export (Phase 5, wave 2)

First read and follow `scripts/agents/prompts/_rules.md`. Then read `docs/STATUS.md` and the Phase 5 section of `docs/EXECUTION_PLAN.md` (including "Rules for the UI").

## Your files
- `src/vaultmind/ui/chat.py`, `src/vaultmind/ui/trace_view.py`, `src/vaultmind/ui/citation_view.py`: implement the stubs, keep the public functions.
- `tests/test_ui_chat.py` (new).

## Contract (read, don't edit)
- `ui/app.py` (shell: header, routing, `_when_ready` so `page()` only runs once the engine is ready) and `ui/state.py` (`state().engine`, `state().cfg`). `tests/test_ui_shell.py` shows how to drive pages in tests with `nicegui.testing.user_simulation(root=...)` and `@pytest.mark.anyio`.
- `app/engine.py`: `Engine.ask(chat_id, question)` is a **blocking** generator: answer tokens (`str`), then the stored assistant `ChatMessage`. It raises on errors (e.g. `OllamaUnavailableError` from `core/ollama.py`) and then stores nothing. `engine.chats` (`store/chats.py`: `recent`, `create`, `rename`, `delete`, `messages`), `engine.traces.get(run_id)` (`store/traces.py`, gives `events`).
- `app/export.py`: `answer_markdown`, `chat_markdown`, `markdown_pdf`. `app/highlight.py`: `render_pdf_citation`, `highlighted_pdf_copy`, `find_passage`. `agent/trace_text.py`: `summarize(node, data)`, `format_trace`. `agent/baseline.py`: `Source` (`label`, `file_path`, `page`, `text`, `warnings`), `cited_sources`.

## Chat page (`page()`)
- Left drawer: "New chat" button and the chat list (`chats.recent()`), current chat highlighted; rename and delete per chat (delete asks for confirmation). Open the most recent chat on load, or an empty new one. Keep the open chat across page navigations (e.g. `app.storage.client` or a module-level selection per client).
- Messages: user and assistant messages (`ui.chat_message` or cards). The answer renders as Markdown. Hebrew answers and questions render right-to-left (`dir="auto"` on the text elements).
- Under each answer:
  - citation chips for the cited sources (`cited_sources(answer.content, answer.sources)`, or all sources if none cited); on abstention, a "Closest matches" row instead. Each chip shows `[n] file, p. N`, has the full label as tooltip, and opens `open_citation(source)`. A source with `warnings` shows a warning icon and the warning text in its tooltip.
  - contradictions ("Sources disagree") and unsupported claims, as a highlighted note.
  - a collapsed "Trace" expansion that loads `engine.traces.get(run_id)` only when opened, then calls `trace_view(events)`.
  - export actions: answer as Markdown or PDF (`ui.download(bytes, filename)`; downloads are enabled in the native window).
- Chat header: export the whole chat as Markdown or PDF.
- Input: textarea at the bottom. Enter sends, Shift+Enter adds a newline. While answering, the input and the send button are disabled and a spinner shows.
- Streaming: never block the event loop. Drive the blocking generator from a worker thread (e.g. `await run.io_bound(next, it, sentinel)` per token, or a thread feeding a queue drained by a short `ui.timer`) and append the tokens to the answer element as they arrive. On the final `ChatMessage`, re-render that message with chips, notes, trace and export. On an exception: `ui.notify` it (`type="negative"`), remove the partial answer, and put the question back in the input.
- New chat: the first exchange titles it (the store does that); refresh the drawer after each answer.

## Trace view (`trace_view(events)`)
- A compact timeline, one row per event: node name, duration (ms), the `summarize` text. Total time at the bottom.
- The `grade` event gets a small table of its `scores` (label, score as a bar or number, passed/failed), sorted by score.

## Citation viewer (`open_citation(source)`)
- A dialog titled with `source.label`.
- PDF with a page: render with `render_pdf_citation` off the event loop (`run.io_bound`), show the PNG (`ui.image` with a data URL or bytes), say "Passage not found on the page; showing the page" when `highlighted` is False. Buttons: previous/next page (re-render that page without a passage, or with it; keep it simple) and "Open in viewer" (`highlighted_pdf_copy` into a temp folder under `state().cfg.paths.data_dir / "tmp"`, then open it with the OS default app: `os.startfile` on Windows, `open` on macOS, `xdg-open` on Linux).
- Markdown/text: show the file text with the passage span from `find_passage` marked (escape the HTML, wrap the span in `<mark>`, scroll it into view). If not found, show the passage text alone with a note.
- A missing file or a render error shows a readable message in the dialog, never a stack trace.

## Tests (no models)
- Build an `Engine` with fakes: `FakeLLM` (responses that cite `[1]`), `FakeEmbedder`, `retrieval.rerank: false` (no reranker), a small real index made with `ingest_path` over a couple of temp Markdown files, real `Registry`/`TraceStore`/`ChatStore` in `tmp_path`. Set `ui_state.init_state(cfg).engine = engine` and `.phase = "ready"`.
- With `user_simulation`: ask a question → the answer text and a citation chip appear; the chat appears in the drawer with its auto title; a new chat starts empty; switching chats shows its messages; an LLM that raises `OllamaUnavailableError` → an error notification and the question back in the input; the trace expansion shows node names; `trace_view` on hand-made events shows the grade scores. Test `open_citation` on a Markdown source (the marked passage appears). Keep the file under ~20 s.
