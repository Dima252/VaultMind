# Task: Ollama providers

First read and follow `scripts/agents/prompts/_rules.md`.

## Your files
- `src/vaultmind/core/ollama.py` (implement the stubs)
- `tests/test_ollama.py` (new, unit tests)
- `tests/test_ollama_live.py` (new, `@pytest.mark.ollama` integration tests)

## Goal
Implement `OllamaLLM` and `OllamaEmbedder` so they satisfy the `LLM` and `Embedder` protocols in `src/vaultmind/core/providers.py`. Use the official `ollama` Python client (`ollama.Client(host=...)`). Check the installed version's API in `.venv` rather than assuming it.

## Requirements
- `from_config(cfg)` reads `cfg.models`, `cfg.ollama`. Set `model_name` to the model tag.
- **`generate`**: `client.chat(...)` with `options={"num_ctx": ...}` and `keep_alive` on **every** call; Ollama's default context silently truncates.
  - With `schema`, pass it as `format=schema` and use `temperature: 0`. Return the raw JSON string.
  - Qwen3 models "think" by default. For schema calls, disable thinking if the installed client supports it (`think=False`). Never return thinking text as content.
- **`stream`**: `chat(..., stream=True)`, yield only non-empty content deltas.
- **`embed`**: `client.embed(model=..., input=batch, keep_alive=...)` in batches of `batch_size`, preserving order. Return `[]` for empty input without calling the server.
- **Errors**: define `OllamaUnavailableError(RuntimeError)` in your module. Raise it, with an actionable message ("is Ollama running at <host>?" / "run: ollama pull <model>"), when the connection fails or the model is missing. Don't swallow other errors.
- Add an optional keyword-only `client` parameter to both constructors so tests can inject a fake client.

## Tests
- Unit (`tests/test_ollama.py`): a fake client object records calls. Assert `num_ctx`/`keep_alive` are always sent, `format` and `temperature` are set only with a schema, batching splits and keeps order, empty-input embed makes no call, connection errors map to `OllamaUnavailableError`, and streaming skips empty deltas.
- Live (`tests/test_ollama_live.py`, marked `ollama`): generate a short reply; a schema call returns parseable JSON matching the schema; embed returns vectors of equal length for 3 texts. These are skipped by default (`addopts` excludes the marker).
