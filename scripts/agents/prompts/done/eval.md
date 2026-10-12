# Task: Evaluation dataset + metrics

First read and follow `scripts/agents/prompts/_rules.md`.

## Your files
- `src/vaultmind/eval/dataset.py` (implement the stubs)
- `src/vaultmind/eval/metrics.py` (implement the stubs)
- `eval/datasets/example.jsonl` (new: 5 illustrative items, generic content, not real user documents)
- `tests/test_eval_dataset.py`, `tests/test_eval_metrics.py` (new)

## Goal
The foundation of the Phase 2 evaluation harness. These numbers will appear in the README and on a CV, so the metrics must be correct, documented, and fully tested.

## Dataset
- `load_dataset`: JSONL, UTF-8. Skip blank lines. On invalid JSON or a schema error, raise `ValueError` naming the file and **line number**.
- `save_dataset`: one compact JSON object per line, UTF-8, `ensure_ascii=False`, trailing newline. Create parent directories. Writing then loading must round-trip exactly.
- Add a pydantic validator to `EvalItem`: if `answerable` is false, then `source_file` and `source_page` must be None. If `answerable` is true, `source_file` is required. `source_page`, when given, is >= 1.

## Metrics
- `source_matches(retrieved, gold)`: compare file names after normalizing (forward slashes, case-insensitive, strip a leading `./`). Match when the paths are equal or when one is just a bare file name equal to the other's basename. A gold page of `None` matches any page; otherwise pages must be equal.
- `recall_at_k`, `mrr`: inputs are answerable items only (document this). A hit is any retrieved source among the first `k` (or anywhere, for MRR) that matches gold. Duplicate retrieved sources don't count twice.
- `abstention_accuracy`: correct means abstained == (not answerable).
- `percentile`: linear interpolation (same as numpy's default), q in [0, 100].
- All functions: raise `ValueError` on empty inputs, length mismatches, `k < 1`, or q out of range. No numpy; the standard library is enough.
- Docstrings should state each formula precisely. The README will cite them.

## Tests
Hand-computed expected values for every metric, including edge cases: gold page None, path normalization variants, a hit exactly at rank k and just past k, duplicates, all-correct and all-wrong abstention, percentile at 0, 50, 95, 100 and with a single value. For datasets: round-trip (including non-ASCII text such as Hebrew), the line number appears in errors, blank lines are skipped, the validator rules hold, and `example.jsonl` loads.
