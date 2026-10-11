import json
import zipfile
from pathlib import Path

import pytest

from vaultmind.eval.cuad import (
    QUESTIONS,
    Contract,
    build_items,
    contract_name,
    import_cuad,
    load_contracts,
    select_contracts,
)
from vaultmind.eval.dataset import load_dataset

CATEGORIES = list(QUESTIONS)


def _doc(title: str, text: str, present: dict[str, list[str]]) -> dict[str, object]:
    qas = [
        {
            "id": f"{title}__{cat}",
            "question": f"Highlight the parts related to {cat}",
            "answers": [{"text": s, "answer_start": text.find(s)} for s in present.get(cat, [])],
            "is_impossible": cat not in present,
        }
        for cat in [*CATEGORIES, "Document Name"]
    ]
    return {"title": title, "paragraphs": [{"context": text, "qas": qas}]}


def _write_zip(path: Path, n: int = 6) -> Path:
    docs = []
    for i in range(n):
        title = f"ACME{i}CORP_01_02_200{i}-EX-10.{i}-SUPPLY AGREEMENT"
        text = f"SUPPLY AGREEMENT {i}\r\nGoverned by the laws of Delaware.\nParty A{i} and B{i}."
        present = {
            "Governing Law": ["laws of Delaware"],
            "Parties": [f"Party A{i}", f"  Party   A{i} ", f"B{i}"],
        }
        docs.append(_doc(title, text, present))
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("CUADv1.json", json.dumps({"version": "aok_v1.0", "data": docs}))
    return path


# --- names ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "name"),
    [
        (
            "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT",
            "Limeenergyco Distributor Agreement (1999)",
        ),
        (
            "LohaCompanyltd_20191209_F-1_EX-10.16_11917878_EX-10.16_Supply Agreement",
            "LohaCompanyltd Supply Agreement (2019)",
        ),
        (
            "PfHospitalityGroupInc_20150923_10-12G_EX-10.1_9266710_EX-10.1_Franchise Agreement3",
            "PfHospitalityGroupInc Franchise Agreement (2015)",
        ),
        ("MetLife, Inc. - Remarketing Agreement", "MetLife, Inc. Remarketing Agreement"),
        ("Plain Title", "Plain Title"),
    ],
)
def test_contract_name(title: str, name: str) -> None:
    assert contract_name(title) == name


def test_contract_name_is_file_safe_and_bounded() -> None:
    name = contract_name("A_B_01_02_2003-EX-1-" + "VERY LONG AGREEMENT: X/Y? " * 10)
    assert not set(name) & set('<>:"/\\|?*')
    assert len(name) < 100


# --- loading and selection --------------------------------------------------


def test_load_contracts(tmp_path: Path) -> None:
    contracts = load_contracts(_write_zip(tmp_path / "data.zip", n=2))
    assert len(contracts) == 2
    first = contracts[0]
    assert first.clauses["Governing Law"] == ["laws of Delaware"]
    assert first.clauses["Non-Compete"] == []
    assert "Document Name" in first.clauses


def test_select_is_seeded_and_filters_length() -> None:
    contracts = [Contract(f"T{i}", "x" * (i * 10), {}) for i in range(10)]
    a = select_contracts(contracts, 3, seed=1, max_chars=50)
    assert a == select_contracts(contracts, 3, seed=1, max_chars=50)
    assert all(len(c.text) <= 50 for c in a)
    assert [c.title for c in a] == sorted(c.title for c in a)
    with pytest.raises(ValueError, match="only 6"):
        select_contracts(contracts, 7, seed=1, max_chars=50)


# --- items ------------------------------------------------------------------


def _triples(n: int = 4) -> list[tuple[str, str, Contract]]:
    clauses: dict[str, list[str]] = {cat: [] for cat in CATEGORIES} | {
        "Governing Law": ["laws of Delaware"],
        "Parties": ["A", "B"],
        "Insurance": ["  general   liability  "],
    }
    return [(f"cuad/C{i}.txt", f"C{i} Agreement", Contract(f"C{i}", "", clauses)) for i in range(n)]


def test_build_items_counts_and_fields() -> None:
    items = build_items(_triples(), n_questions=10, unanswerable_share=0.2, seed=0)
    assert len(items) == 10
    unanswerable = [i for i in items if not i.answerable]
    assert len(unanswerable) == 2
    for item in unanswerable:
        assert item.source_file is None and item.evidence == [] and item.answer == ""
    for item in items:
        assert item.tags[0] == "cuad"
        assert item.question.count("the C") == 1  # names its contract
        if item.answerable:
            assert item.source_file is not None and item.source_file.startswith("cuad/C")
            assert item.evidence and item.answer == " ... ".join(item.evidence)
    insurance = [i for i in items if "insurance" in i.tags and i.answerable]
    for item in insurance:
        assert item.evidence == ["general liability"]


def test_build_items_spreads_categories() -> None:
    items = build_items(_triples(), n_questions=6, unanswerable_share=0.0, seed=3)
    # Three answerable categories, so six picks cover each one twice.
    tags = sorted(i.tags[1] for i in items)
    assert tags == ["governing-law"] * 2 + ["insurance"] * 2 + ["parties"] * 2


def test_related_category_blocks_unanswerable() -> None:
    clauses: dict[str, list[str]] = {cat: [] for cat in CATEGORIES} | {
        "Non-Compete": ["will not endorse competing products"],  # CUAD's home for exclusivity
        "Governing Law": ["laws of Delaware"],
    }
    triples = [("cuad/C.txt", "C", Contract("C", "", clauses))]
    n_empty = len(CATEGORIES) - 3  # minus the two labeled categories and blocked Exclusivity
    items = build_items(triples, n_empty + 2, unanswerable_share=n_empty / (n_empty + 2), seed=0)
    unanswerable = {i.tags[1] for i in items if not i.answerable}
    assert len(unanswerable) == n_empty
    assert "exclusivity" not in unanswerable
    assert "insurance" in unanswerable


def test_build_items_is_deterministic() -> None:
    a = build_items(_triples(), 8, 0.25, seed=5)
    assert a == build_items(_triples(), 8, 0.25, seed=5)


def test_build_items_rejects_too_many() -> None:
    with pytest.raises(ValueError, match="answerable"):
        build_items(_triples(1), n_questions=10, unanswerable_share=0.0, seed=0)


# --- end to end -------------------------------------------------------------


def test_import_cuad(tmp_path: Path) -> None:
    zip_path = _write_zip(tmp_path / "data.zip")
    vault = tmp_path / "vault"
    stale = vault / "cuad" / "old contract.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("old", encoding="utf-8")
    keep = vault / "mine.txt"
    keep.write_text("user file", encoding="utf-8")
    dataset = tmp_path / "ds.jsonl"

    summary = import_cuad(
        zip_path, vault, dataset, n_contracts=3, n_questions=5, unanswerable_share=0.4, seed=0
    )

    assert len(summary.written) == 3
    assert summary.removed == [stale] and not stale.exists()
    assert keep.exists()
    raw = summary.written[0].read_bytes()
    assert b"\r\n" in raw  # text is written exactly as CUAD has it
    items = load_dataset(dataset)
    assert items == summary.items
    files = {f"cuad/{p.name}" for p in summary.written}
    assert all(i.source_file in files for i in items if i.answerable)
    assert sum(not i.answerable for i in items) == 2
