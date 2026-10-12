# Task: Task eval set, key-facts judge, repeated runs (Phase 5.5, M1c)

First read and follow `scripts/agents/prompts/_rules.md`. Then read `docs/STATUS.md`, `docs/RESULTS.md` (how numbers are reported, and the ±5-point judge noise) and the Phase 5.5 section of `docs/EXECUTION_PLAN.md` (the "Eval" task is yours, except running the agent on it: the planner doesn't exist yet).

## Your files
- `src/vaultmind/eval/judge.py`: add the key-facts judge. Keep everything that's there.
- `src/vaultmind/eval/metrics.py`: add the new metrics.
- `src/vaultmind/eval/runner.py` and `src/vaultmind/cli.py`: `eval run --repeat N` only. Touch nothing else in `cli.py`.
- `tests/test_eval_judge.py`, `tests/test_eval_metrics.py`, `tests/test_eval_runner.py`, `tests/test_cli.py`: add tests.
- New: `eval/datasets/tasks_example.jsonl` (synthetic, committed).
- Outside git: `C:/Users/dimat/Documents/GitHub/VaultMind/eval/datasets/personal_tasks.jsonl` and `personal_tasks_review.md` (gitignored; never in your worktree, never committed).

## Contract (read, don't edit)
- `eval/tasks.py`: `TaskItem`, `load_tasks`, `save_tasks`. The format is fixed. Build items with the model and write them with `save_tasks` so they validate.
- `agent/types.py`: `TaskKind`, `TASKS`.

## 1. Key-facts judge (`eval/judge.py`)
`judge_key_facts(request: str, key_facts: Sequence[str], answer: str, *, llm: LLM) -> KeyFactsResult` with `KeyFactsResult(found: tuple[bool, ...], reasons: tuple[str, ...])` and a `score` property (fraction found; 1.0 for no facts). One schema call returns, per numbered fact, whether the answer states it (paraphrase and other languages count: a Hebrew answer can state an English fact) and a short reason. Facts the answer contradicts count as not found. Same retry and `JudgeError` behavior as `judge()`. Empty answer: all false, no call. Also `judge_clarify(request, answer, *, llm) -> bool`: did the response ask the user which document or what they meant, rather than answer? Prompts are module constants (they'll be quoted in RESULTS.md); the answer is data inside delimiters.

## 2. Metrics (`eval/metrics.py`)
`router_accuracy(predicted, gold)` and `router_confusion(predicted, gold) -> dict[str, dict[str, int]]`; `task_success(scores, threshold=0.75)` (share of items whose key-facts score is at least the threshold); `mean_and_spread(values_per_repeat) -> (mean, min, max)` for repeated runs. Docstrings state exact formulas, like the existing ones.

## 3. `eval run --repeat N`
Runs every (config, dataset) pair N times (default 1) and reports each metric as the mean with its min-max range when N > 1 (the table gets a ± column or "0.80 (0.77-0.83)"). Raw results are saved per repeat (`<config>__<dataset>__r1.jsonl`, ...); N = 1 keeps today's file names exactly, so `eval sweep` and old results still work.

## 4. The task set
**About 60 items** over the main checkout's vault (`vault/personal/`: the user's 12 files, mostly Hebrew course material plus some English Workato PDFs; `vault/cuad/`: 25 English contracts; `vault/adversarial/`). About half the requests in Hebrew. Mix: list 5, summarize 10, compare 8, extract 8, transform 8 (with `history` holding the previous answer; "make it shorter", "translate that to English", "as a table"), open multi-step 15 ("which course has the heavier workload, and why?", "find the termination clause in X, then tell me how many days notice that is in weeks"), clarify 6 (genuinely ambiguous: two syllabi, "the contract"). Add 2-3 `injection` items over `vault/adversarial/` where a document tries to make the agent call a tool or reveal something (`forbidden` lists the payload codes).
- Read the documents to write `key_facts`: 2-5 short, checkable facts each, in the language the answer will use. Facts must be true and specific (a number, a name, a date, which document has more of something), not vague ("mentions grading").
- `documents` lists the files a correct agent reads (vault-relative, e.g. `personal/<file>.pdf`).
- Write it to the main checkout's `eval/datasets/personal_tasks.jsonl` (gitignored) and a `personal_tasks_review.md` next to it: one row per item with id, task, request, key facts and where you found each (file, page), so the user can verify them by hand as they did for `personal.jsonl`.
- **Privacy:** the repo is public. Personal content goes only in those two gitignored files. Your commit, code, tests and summary describe the set by counts and types only.
- `eval/datasets/tasks_example.jsonl`: ~8 synthetic items (no real document content) that exercise every field, for tests and as format documentation.

## Tests (no models)
The judge with `FakeLLM` (each case as for `judge()`, plus mixed found/not-found, length mismatch in the model's output treated as unparseable, clarify yes/no); the metrics with hand-computed values; `--repeat` with fake configs (file names for N = 1 and N = 2, mean/min/max in the table); `tasks_example.jsonl` loads.

## Measure (live, at the end)
Check the key-facts judge against 10 hand-made (request, facts, answer) cases with known truth, half in Hebrew; report agreement. Run it 3 times to see whether verdicts are stable.
