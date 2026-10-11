# Task: Eval runner (Phase 2)

First read and follow `scripts/agents/prompts/_rules.md`. Then read `docs/STATUS.md` for context.

## Your files
- `src/vaultmind/eval/runner.py` (implement the stubs; keep every public signature and dataclass)
- `tests/test_eval_runner.py` (new)

## Goal
One function call scores a config on a dataset. These numbers go into `docs/RESULTS.md`, the README, and a CV, so they must be correct, deterministic, and fully tested.

## Use, don't reimplement
- Retrieval and prompting: `vaultmind.agent.baseline` (`retrieve`, `build_messages`, `is_abstention`, `NOT_FOUND`).
- Metrics: `vaultmind.eval.metrics` (`source_matches` via `recall_at_k`/`mrr`, `evidence_recall_at_k`, `evidence_mrr`, `abstention_accuracy`, `injection_resistance`, `correctness_score`, `percentile`). Read their docstrings for input rules (for example, recall takes answerable items only).
- Judge: `vaultmind.eval.judge.judge(question, reference, answer, *, llm)`. It is a stub in your worktree (another agent is writing it), so **in tests, monkeypatch `vaultmind.eval.runner.judge`** (import it as a module-level name in `runner.py` so it can be patched). Catch `JudgeError`.

## `run_config`
For each item, in dataset order:
1. Time `retrieve(...)` with `time.perf_counter()` -> `retrieval_ms`.
2. If sources were found, time `llm.generate(build_messages(question, sources))` (strip the result) -> `generation_ms`. Otherwise the answer is `NOT_FOUND` and `generation_ms` is 0.
3. `abstained = is_abstention(answer)`.
4. `RetrievedSource.file`: the source path relative to `cfg.paths.vault_dir.resolve()`, with forward slashes; the absolute path (forward slashes) if it's outside the vault; `""` if the path is unknown. `page`, `text` (the passage text), `score` come from the baseline `Source`.
5. Verdicts follow the comments on `ItemResult` exactly:
   - judge off (`judge_llm is None`): verdict and reason are None for every item.
   - judge on: unanswerable item -> None/None. Answerable and abstained -> `"incorrect"` with reason None and **no judge call**. Answerable and answered -> `judge(item.question, item.answer, answer, llm=judge_llm)`; store `verdict.value` and `reason`. On `JudgeError`: verdict None, reason `"judge error: <message>"`.
6. Call `on_item(result)` after each item if given.
Then `metrics = compute_metrics(items, results)`.

## `compute_metrics`
Returns a dict with **exactly** the keys of `METRIC_NAMES`, in order. A metric is None when no item qualifies (never raise for that).
- `recall@1`, `recall@5`, `mrr`: answerable items. Gold = `(source_file, source_page)`; retrieved = `[(r.file, r.page) ...]`.
- `evidence_recall@1`, `evidence_recall@5`, `evidence_mrr`: items with non-empty `evidence`; passages = `[r.text ...]`.
- `correctness`: `correctness_score` over results whose verdict is not None.
- `abstention_accuracy`: all items.
- `injection_resistance`: items with non-empty `forbidden`, using the answer text.
- Latency p50/p95: `percentile` over all items; total = retrieval + generation.
Raise ValueError if `items` and `results` differ in length.

## `check_compatible`
Group configs by `paths.data_dir.resolve()`. Within a group, raise ValueError naming both config names if `models.embedder` or any `chunking` setting differs. Different data_dirs are always fine.

## `save_report`
`<out_dir>/<config_name>__<dataset_name>.jsonl`, create directories. One line per `ItemResult` (`dataclasses.asdict`), then one final line `{"metrics": {...}}`. UTF-8, `ensure_ascii=False`, compact separators, `\n` line endings. Sanitize names so the file name is valid on Windows.

## `format_table`
ASCII only (the dev console is cp1255). First column: metric names in `METRIC_NAMES` order. One column per report, headed `config / dataset`. Ratios with 3 decimals (`0.812`), `*_ms` as whole numbers, None as `-`. Columns aligned. Add a line under the header. Raise ValueError on an empty list.

## Tests (fakes only, no Ollama)
- Build a real `ChunkIndex` + `Registry` in `tmp_path` with `FakeEmbedder`, the way `tests/test_baseline.py` does, with the registry paths under a `tmp_path` vault dir (set `cfg.paths.vault_dir` with `Config.model_copy(update=...)` or by constructing a `Config`). Use `FakeLLM` with scripted answers.
- Cover: relative paths (inside and outside the vault), NOT_FOUND when retrieval is empty (generation_ms 0, no LLM call), every verdict rule (including no judge call on abstention, JudgeError, judge off), `on_item` calls, hand-computed metrics on a small mixed set (answerable with and without evidence, unanswerable, an injection item), None metrics when nothing qualifies, exact metric key order, `check_compatible` pass/fail cases, `save_report` round-trip (read the lines back), and `format_table` content and alignment (and that it is pure ASCII).
- Latency can't be asserted exactly. Monkeypatch `time.perf_counter` in the runner module if you want deterministic values.
