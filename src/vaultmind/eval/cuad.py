"""Import a CUAD subset as an eval corpus + dataset.

CUAD (https://www.atticusprojectai.org/cuad, CC BY 4.0) is 510 commercial contracts in SQuAD 2.0
format, with clause spans labeled by lawyers for 41 categories. We pick a seeded subset of
contracts, write each one as a .txt file under ``<vault>/cuad/``, and turn some of their labels
into eval items. Questions name their contract so they stay unambiguous in a multi-document
vault. A category with no labeled span becomes an unanswerable item: that clause is absent from
that contract (the annotators' call, so spot-check those items).
"""

from __future__ import annotations

import json
import random
import re
import zipfile
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vaultmind.eval.dataset import EvalItem, save_dataset

CUAD_JSON = "CUADv1.json"
SUBDIR = "cuad"

# Category -> question. Only categories whose absence reads as "not in this contract" are used.
QUESTIONS: dict[str, str] = {
    "Parties": "Who are the parties to {contract}?",
    "Agreement Date": "On what date was {contract} made?",
    "Expiration Date": "When does {contract} expire?",
    "Renewal Term": "How does {contract} renew after its initial term?",
    "Notice Period To Terminate Renewal": (
        "How much notice is required to prevent {contract} from renewing?"
    ),
    "Governing Law": "Which jurisdiction's law governs {contract}?",
    "Most Favored Nation": "What most-favored-nation terms does {contract} include?",
    "Non-Compete": "What non-compete restrictions does {contract} impose?",
    "Exclusivity": "What exclusivity commitments does {contract} contain?",
    "No-Solicit Of Employees": (
        "What does {contract} say about soliciting or hiring the other party's employees?"
    ),
    "Termination For Convenience": (
        "Under what terms can {contract} be terminated for convenience, without cause?"
    ),
    "Change Of Control": "What happens under {contract} if a party undergoes a change of control?",
    "Anti-Assignment": "What restrictions does {contract} place on assigning it?",
    "Revenue/Profit Sharing": "What revenue or profit sharing does {contract} require?",
    "Minimum Commitment": "What minimum purchase or volume commitment does {contract} require?",
    "Ip Ownership Assignment": "Who owns the intellectual property created under {contract}?",
    "License Grant": "What license does {contract} grant?",
    "Audit Rights": "What audit or inspection rights does {contract} give?",
    "Cap On Liability": "How is liability capped under {contract}?",
    "Liquidated Damages": "What liquidated damages or termination fees does {contract} set?",
    "Warranty Duration": "How long do the warranties under {contract} last?",
    "Insurance": "What insurance must be maintained under {contract}?",
    "Source Code Escrow": "What source code escrow arrangement does {contract} set up?",
}

# EDGAR-style titles: FILER_YYYYMMDD_..._Kind or FILER_MM_DD_YYYY-EX-n-KIND.
_EDGAR = re.compile(r"^(?P<filer>.+?)_(?P<date>\d{8}|\d\d_\d\d_\d{4})[-_](?P<rest>.+)$")
_MAX_KIND = 60  # some titles carry a whole sentence ("... AGREEMENT between X and Y")
_UNSAFE = re.compile(r'[<>:"/\\|?*_]+')  # not allowed in Windows file names (and "_")


@dataclass(frozen=True)
class Contract:
    title: str
    text: str
    clauses: dict[str, list[str]]  # category -> labeled spans; empty when the clause is absent


@dataclass
class ImportSummary:
    written: list[Path]
    removed: list[Path]
    items: list[EvalItem]


def load_contracts(zip_path: Path) -> list[Contract]:
    """Read every contract and its labels from CUAD's ``data.zip``."""
    with zipfile.ZipFile(zip_path) as z:
        data: dict[str, Any] = json.loads(z.read(CUAD_JSON))
    contracts = []
    for doc in data["data"]:
        para = doc["paragraphs"][0]
        clauses = {
            qa["id"].rsplit("__", 1)[1]: [a["text"] for a in qa["answers"]] for qa in para["qas"]
        }
        contracts.append(Contract(title=doc["title"], text=para["context"], clauses=clauses))
    return contracts


def _tidy(text: str) -> str:
    text = " ".join(_UNSAFE.sub(" ", text).split())
    return text.title() if text.isupper() else text


def contract_name(title: str) -> str:
    """Readable, file-name-safe name for a CUAD title.

    ``LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT`` -> ``Limeenergyco Distributor
    Agreement (1999)``; ``MetLife, Inc. - Remarketing Agreement`` -> ``MetLife, Inc. Remarketing
    Agreement``. Not guaranteed unique; see ``_unique_names``.
    """
    title = title.strip()
    year = None
    if m := _EDGAR.match(title):
        party, date, rest = m["filer"], m["date"], m["rest"]
        if "_" in date:
            year, kind = date[-4:], re.sub(r"^EX-[^-]*-", "", rest)
        else:
            year, kind = date[:4], rest.rsplit("_", 1)[-1]
    elif " - " in title:
        party, kind = title.split(" - ", 1)
    else:
        party, kind = "", title
    kind = _tidy(kind.rstrip("0123456789 "))  # "Franchise Agreement3"
    if len(kind) > _MAX_KIND:
        kind = kind[: _MAX_KIND + 1].rsplit(" ", 1)[0]
    kind = kind.rstrip(".,;:-~ ")
    name = " ".join(part for part in (_tidy(party), kind) if part)
    return f"{name} ({year})" if year else name


def _unique_names(contracts: Sequence[Contract]) -> list[str]:
    seen: dict[str, int] = {}
    names = []
    for c in contracts:
        base = contract_name(c.title)
        seen[base.lower()] = seen.get(base.lower(), 0) + 1
        count = seen[base.lower()]
        names.append(base if count == 1 else f"{base} #{count}")
    return names


def select_contracts(
    contracts: Sequence[Contract], n: int, seed: int, max_chars: int
) -> list[Contract]:
    """A seeded sample of ``n`` contracts no longer than ``max_chars``, sorted by title."""
    eligible = sorted((c for c in contracts if len(c.text) <= max_chars), key=lambda c: c.title)
    if n > len(eligible):
        raise ValueError(f"asked for {n} contracts, only {len(eligible)} have <= {max_chars} chars")
    return sorted(random.Random(seed).sample(eligible, n), key=lambda c: c.title)


def _slug(category: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", category.lower()).strip("-")


def _spans(raw: Sequence[str]) -> list[str]:
    """Whitespace-collapsed, non-empty, de-duplicated spans in their original order."""
    spans = [" ".join(s.split()) for s in raw]
    return list(dict.fromkeys(s for s in spans if s))


_Candidate = tuple[str, str, str, list[str]]  # (source_file, name, category, spans)


def _spread(candidates: list[_Candidate], n: int, rng: random.Random) -> list[_Candidate]:
    """Pick ``n`` candidates round-robin across categories, so common clauses don't dominate."""
    by_category: dict[str, list[_Candidate]] = defaultdict(list)
    for cand in candidates:
        by_category[cand[2]].append(cand)
    queues = [by_category[cat] for cat in QUESTIONS if cat in by_category]
    for queue in queues:
        rng.shuffle(queue)
    rng.shuffle(queues)
    picked: list[_Candidate] = []
    while len(picked) < n:
        for queue in queues:
            if queue and len(picked) < n:
                picked.append(queue.pop())
    return picked


def build_items(
    contracts: Sequence[tuple[str, str, Contract]],
    n_questions: int,
    unanswerable_share: float,
    seed: int,
) -> list[EvalItem]:
    """Eval items from ``(source_file, name, contract)`` triples.

    ``round(n_questions * unanswerable_share)`` items are unanswerable; the rest are answerable,
    with the labeled spans as ``evidence`` and joined with `` ... `` as the reference answer.
    """
    if not 0.0 <= unanswerable_share <= 1.0:
        raise ValueError(f"unanswerable_share must be in [0, 1], got {unanswerable_share}")
    answerable: list[_Candidate] = []
    unanswerable: list[_Candidate] = []
    for source_file, name, contract in contracts:
        for category in QUESTIONS:
            if category not in contract.clauses:
                continue
            spans = _spans(contract.clauses[category])
            (answerable if spans else unanswerable).append((source_file, name, category, spans))

    n_unanswerable = round(n_questions * unanswerable_share)
    n_answerable = n_questions - n_unanswerable
    if n_answerable > len(answerable) or n_unanswerable > len(unanswerable):
        raise ValueError(
            f"need {n_answerable} answerable + {n_unanswerable} unanswerable questions, "
            f"have {len(answerable)} + {len(unanswerable)}; use more contracts or fewer questions"
        )
    rng = random.Random(seed)
    picked = _spread(answerable, n_answerable, rng) + _spread(unanswerable, n_unanswerable, rng)
    rng.shuffle(picked)
    return [
        EvalItem(
            question=QUESTIONS[category].format(contract=f"the {name}"),
            answer=" ... ".join(spans),
            source_file=source_file if spans else None,
            source_page=None,
            answerable=bool(spans),
            tags=["cuad", _slug(category)],
            evidence=spans,
        )
        for source_file, name, category, spans in picked
    ]


def import_cuad(
    zip_path: Path,
    vault_dir: Path,
    dataset_path: Path,
    *,
    n_contracts: int = 25,
    n_questions: int = 60,
    unanswerable_share: float = 0.2,
    seed: int = 0,
    max_chars: int = 100_000,
) -> ImportSummary:
    """Write the sampled contracts to ``<vault_dir>/cuad/`` and their questions to ``dataset_path``.

    ``<vault_dir>/cuad/`` belongs to the importer: ``.txt`` files there that aren't in the new
    sample are removed, so a different seed never leaves extra contracts behind as distractors.
    """
    chosen = select_contracts(load_contracts(zip_path), n_contracts, seed, max_chars)
    out_dir = vault_dir / SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    triples: list[tuple[str, str, Contract]] = []
    for contract, name in zip(chosen, _unique_names(chosen), strict=True):
        path = out_dir / f"{name}.txt"
        data = contract.text.encode("utf-8")
        if not path.exists() or path.read_bytes() != data:
            path.write_bytes(data)
        written.append(path)
        triples.append((f"{SUBDIR}/{path.name}", name, contract))

    keep = {p.name for p in written}
    removed = sorted(p for p in out_dir.glob("*.txt") if p.name not in keep)
    for path in removed:
        path.unlink()

    items = build_items(triples, n_questions, unanswerable_share, seed)
    save_dataset(items, dataset_path)
    return ImportSummary(written=written, removed=removed, items=items)
