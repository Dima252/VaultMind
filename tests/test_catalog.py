"""Document catalog: entries, resolving a mention to one file, the planner's description."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from vaultmind.agent.catalog import Catalog, first_sentence
from vaultmind.agent.types import AmbiguousDocument, DocumentNotFound
from vaultmind.core.models import FileRecord, file_id_for
from vaultmind.store.cards import CardStore
from vaultmind.store.registry import Registry

from .test_tools import DOCS, HEBREW_TITLE, Vault, build_vault

SYLLABUS = DOCS["syllabus"][0]
WORKSHEET = DOCS["worksheet"][0]
CHECKLIST = DOCS["checklist"][0]
NOTES = DOCS["notes"][0]
MEMO = DOCS["memo"][0]


class ScriptedReranker:
    """Scores a passage by the first scripted key it contains (0 when none)."""

    model_name = "scripted"

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.passages: list[str] = []

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.passages = list(passages)
        return [max((s for k, s in self.scores.items() if k in p), default=0.0) for p in passages]


@pytest.fixture
def vault(tmp_path: Path) -> Vault:
    return build_vault(tmp_path)


def catalog(v: Vault, reranker: ScriptedReranker | None = None) -> Catalog:
    return Catalog(v.ctx.registry, v.ctx.cards, v.root, reranker=reranker)


def test_entries_are_ready_files_sorted_with_cards(vault: Vault) -> None:
    entries = vault.ctx.catalog.entries()
    assert [e.name for e in entries] == [SYLLABUS, NOTES, CHECKLIST, MEMO, WORKSHEET]
    with_cards = {e.name for e in entries if e.card is not None}
    assert with_cards == {SYLLABUS, WORKSHEET, NOTES}
    assert entries[0].card is not None and entries[0].card.doc_type == "syllabus"


def test_entries_outside_the_vault_keep_their_absolute_path(tmp_path: Path, vault: Vault) -> None:
    outside = tmp_path / "elsewhere" / "loose.txt"
    outside.parent.mkdir()
    outside.write_text("x", encoding="utf-8")
    vault.ctx.registry.upsert(
        FileRecord(file_id_for(outside), outside.resolve(), "e" * 64, "ready")
    )
    names = [e.name for e in vault.ctx.catalog.entries()]
    assert outside.resolve().as_posix() in names


@pytest.mark.parametrize(
    "mention",
    [
        "Solution design checklist.md",
        "Solution design checklist",
        "SOLUTION DESIGN CHECKLIST",
        '"Solution design checklist"',
    ],
)
def test_resolve_exact_name_or_stem(vault: Vault, mention: str) -> None:
    assert vault.ctx.catalog.resolve(mention).name == CHECKLIST


def test_resolve_by_id_path_and_title(vault: Vault) -> None:
    cat = vault.ctx.catalog
    assert cat.resolve(vault.ids["memo"]).name == MEMO
    assert cat.resolve("courses/AI in the workplace syllabus").name == SYLLABUS
    assert cat.resolve("courses\\AI in the workplace syllabus.pdf").name == SYLLABUS
    assert cat.resolve("AI in the Workplace").name == SYLLABUS  # the card title
    assert cat.resolve(HEBREW_TITLE).name == NOTES


@pytest.mark.parametrize(
    ("mention", "expected"),
    [
        ("the AI syllabus", SYLLABUS),
        ("syllabus", SYLLABUS),
        ("workato worksheet", WORKSHEET),
        ("the Workato worksheeet", WORKSHEET),  # typo
        ("the design checklist", CHECKLIST),
        ("my lecture notes", NOTES),  # card type
        ("vendor memos", MEMO),
        ("the workato recipe sheet", WORKSHEET),  # two of three words
        ("introduction to algorithms", NOTES),  # "intro" in the file name is a prefix
        # "the intro to algorithms" in Hebrew, with the article prefix the title doesn't have
        ("המבוא לאלגוריתמים", NOTES),
    ],
)
def test_resolve_fuzzy(vault: Vault, mention: str, expected: str) -> None:
    assert vault.ctx.catalog.resolve(mention).name == expected


def test_resolve_ambiguous_lists_the_close_candidates(vault: Vault) -> None:
    with pytest.raises(AmbiguousDocument) as info:
        vault.ctx.catalog.resolve("the design doc")
    assert info.value.candidates == (CHECKLIST, WORKSHEET)
    with pytest.raises(AmbiguousDocument) as info:
        vault.ctx.catalog.resolve("the document")
    assert len(info.value.candidates) == 5


def test_resolve_half_a_match_asks_instead_of_guessing(vault: Vault) -> None:
    with pytest.raises(AmbiguousDocument) as info:
        vault.ctx.catalog.resolve("recipe guidelines")
    assert info.value.candidates == (WORKSHEET,)


@pytest.mark.parametrize("mention", ["my 2024 tax return", "", "  "])
def test_resolve_not_found(vault: Vault, mention: str) -> None:
    with pytest.raises(DocumentNotFound):
        vault.ctx.catalog.resolve(mention)


def test_resolve_by_reranker_across_languages(vault: Vault) -> None:
    hebrew = "הסילבוס של הקורס"
    reranker = ScriptedReranker({"AI in the workplace syllabus": 0.9})
    assert catalog(vault, reranker).resolve(hebrew).name == SYLLABUS
    assert len(reranker.passages) == 5 + 3  # every bare name, plus a description per card


def test_reranker_uses_the_card_description_too(vault: Vault) -> None:
    reranker = ScriptedReranker({"A worksheet for designing recipes": 0.8})
    assert catalog(vault, reranker).resolve("integration planning sheet").name == WORKSHEET


def test_reranker_thresholds(vault: Vault) -> None:
    low = ScriptedReranker({"syllabus": 0.1})
    with pytest.raises(DocumentNotFound):
        catalog(vault, low).resolve("course plan")
    close = ScriptedReranker({"syllabus": 0.6, "worksheet": 0.5})
    with pytest.raises(AmbiguousDocument) as info:
        catalog(vault, close).resolve("course plan")
    assert info.value.candidates == (SYLLABUS, WORKSHEET)
    clear = ScriptedReranker({"syllabus": 0.6, "worksheet": 0.3})
    assert catalog(vault, clear).resolve("course plan").name == SYLLABUS


def test_reranker_does_not_override_a_clear_word_match(vault: Vault) -> None:
    reranker = ScriptedReranker({"worksheet": 0.99})
    assert catalog(vault, reranker).resolve("the AI syllabus").name == SYLLABUS
    assert reranker.passages == []


def test_one_document_vault(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "app.sqlite")
    cards = CardStore(tmp_path / "app.sqlite")
    path = tmp_path / "vault" / "only.txt"
    cat = Catalog(registry, cards, tmp_path / "vault")
    with pytest.raises(DocumentNotFound):
        cat.resolve("the document")
    assert cat.describe(10) == "No documents are indexed yet."
    registry.upsert(FileRecord(file_id_for(path), path, "a" * 64, "ready"))
    assert cat.resolve("the document").name == "only.txt"


def test_describe(vault: Vault) -> None:
    lines = vault.ctx.catalog.describe(60).splitlines()
    assert lines[0] == f"{SYLLABUS} | syllabus | en | A course on using AI at work."
    assert lines[2] == CHECKLIST  # no card: the name alone
    assert lines[4] == f"{WORKSHEET} | worksheet | en | A worksheet for designing recipes."
    short = vault.ctx.catalog.describe(2).splitlines()
    assert short[2:] == ["... and 3 more"]
    assert vault.ctx.catalog.describe(0) == "... and 5 more"


def test_first_sentence() -> None:
    assert first_sentence("One thing. Two things.") == "One thing."
    assert first_sentence("Line one\nline two") == "Line one"
    assert first_sentence("version 2.5 is out") == "version 2.5 is out"
    long = first_sentence("x" * 300)
    assert len(long) == 200 and long.endswith("...")


@pytest.mark.ollama
def test_real_reranker_resolves_hebrew_and_english_descriptions(tmp_path: Path) -> None:
    from vaultmind.core.rerank import CrossEncoderReranker

    v = build_vault(tmp_path)
    cat = Catalog(
        v.ctx.registry,
        v.ctx.cards,
        v.root,
        reranker=CrossEncoderReranker("BAAI/bge-reranker-v2-m3"),
    )
    worksheet_he = "דף העבודה לתכנון מתכונים"
    checklist_he = "רשימת הבדיקה לעיצוב פתרון"
    assert cat.resolve(worksheet_he).name == WORKSHEET
    assert cat.resolve(checklist_he).name == CHECKLIST
    assert cat.resolve("the course outline about artificial intelligence").name == SYLLABUS
    with pytest.raises(DocumentNotFound):
        cat.resolve("my 2024 tax return")
