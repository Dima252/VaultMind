"""Run only the planner over task items and show its decisions (no recipes, no judge).

Fast prompt tuning for agent/planner.py: ~2 s per item instead of a full `eval tasks` pass.

    uv run python scripts/dev/plan_probe.py clarify,summarize   # id prefixes
    uv run python scripts/dev/plan_probe.py --lookups            # cuad + personal: stay lookups?
"""

from __future__ import annotations

import json
import sys
from contextlib import closing
from pathlib import Path
from typing import Any

from vaultmind import cli
from vaultmind.agent import planner
from vaultmind.agent.catalog import Catalog
from vaultmind.agent.types import Plan
from vaultmind.core.config import load_config
from vaultmind.core.ollama import OllamaLLM
from vaultmind.store.cards import CardStore

DATASETS = Path("eval/datasets")


def main(arg: str) -> None:
    cfg = load_config(None)
    llm = OllamaLLM.from_config(cfg)
    with cli._store(cfg) as (_, registry), closing(CardStore(cli._db_path(cfg))) as cards:
        catalog = Catalog(registry, cards, cfg.paths.vault_dir, reranker=cli._reranker(cfg))

        def run(question: str) -> Plan:
            return planner.plan(
                question, [], llm=llm, catalog=catalog, catalog_files=cfg.agent.catalog_files
            ).plan

        if arg == "--lookups":
            for name in ("cuad", "personal"):
                rows = _rows(DATASETS / f"{name}.jsonl")
                off = [
                    (p.task, r["question"][:90])
                    for r in rows
                    if (p := run(r["question"])).task != "lookup"
                ]
                print(f"{name}: {len(rows) - len(off)}/{len(rows)} lookup")
                for task, q in off:
                    print(f"    {task:9} {q}")
            return

        items = [
            i
            for i in _rows(DATASETS / "personal_tasks.jsonl")
            if i["id"].startswith(tuple(arg.split(",")))
        ]
        names = {e.file.id: e.name for e in catalog.entries()}
        ok = docs_ok = docs_checked = 0
        for item in items:
            p = run(item["request"])
            want_clarify = bool(item.get("clarify"))
            good = (p.clarify is not None) == want_clarify and (
                want_clarify or p.task == item["task"]
            )
            ok += good
            # Documents: the resolved files against the item's own (when it lists them and
            # doesn't expect a question). None named means the recipe searches for them.
            docs = "-"
            gold = set(item.get("documents") or [])
            if gold and not want_clarify and p.task != "lookup":
                got = {names[d] for d in p.documents}
                docs_checked += 1
                docs_ok += got == gold
                docs = "ok" if got == gold else ("search" if not got else "WRONG")
            print(
                f"{'ok ' if good else 'BAD'} {item['id']:13} task={p.task:9} docs={docs:6} "
                f"n={len(p.documents)} focus={p.focus!r} clarify={(p.clarify or '')[:50]!r}"
            )
        print(f"task/clarify {ok}/{len(items)}, documents {docs_ok}/{docs_checked}")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
