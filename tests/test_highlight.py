"""Highlighted citations: PDF pages rendered with the passage marked, spans in text files."""

import hashlib
from pathlib import Path
from typing import Any

import pymupdf as _pymupdf
import pytest

from vaultmind.app.highlight import (
    _pieces,
    _strip_markdown,
    find_passage,
    highlighted_pdf_copy,
    render_pdf_citation,
)

pymupdf: Any = _pymupdf  # partially typed upstream

PNG = b"\x89PNG\r\n\x1a\n"

PAGE_1 = (
    "Chapter One\n\n"
    "The supplier shall deliver all goods to the warehouse within thirty days of the order. "
    "Late deliveries are subject to a penalty of two percent per week.\n\n"
    "Payment is due within sixty days after the invoice date."
)
PAGE_2 = (
    "Chapter Two\n\n"
    "Either party may terminate this agreement with ninety days written notice to the other "
    "party. Termination does not affect obligations that arose before the notice was given."
)


def _pdf(path: Path, pages: list[str]) -> Path:
    doc: Any = pymupdf.open()
    for text in pages:
        doc.new_page().insert_textbox(pymupdf.Rect(72, 72, 520, 770), text, fontsize=11)
    doc.save(path)
    doc.close()
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def contract(tmp_path: Path) -> Path:
    return _pdf(tmp_path / "contract.pdf", [PAGE_1, PAGE_2])


def test_passage_on_cited_page_is_highlighted(contract: Path) -> None:
    before = _digest(contract)
    passage = (
        "The supplier shall deliver all goods to the warehouse within thirty days of the order."
    )

    image = render_pdf_citation(contract, 1, passage)

    assert image.highlighted
    assert (image.page, image.page_count) == (1, 2)
    assert image.png.startswith(PNG)
    assert _digest(contract) == before  # the user's file is untouched


def test_markdown_markup_is_ignored(contract: Path) -> None:
    passage = (
        "## **Chapter Two**\n\n"
        "Either **party** may _terminate_ this agreement with ninety days written notice to "
        "the other party."
    )
    image = render_pdf_citation(contract, 2, passage)
    assert image.highlighted
    assert image.page == 2


def test_missing_passage_shows_the_cited_page_unmarked(contract: Path) -> None:
    image = render_pdf_citation(
        contract, 2, "Quantum chromodynamics explains how gluons bind quarks inside protons."
    )
    assert not image.highlighted
    assert image.page == 2
    assert image.png.startswith(PNG)


def test_highlighting_changes_the_rendered_page(contract: Path) -> None:
    passage = "Payment is due within sixty days after the invoice date."
    marked = render_pdf_citation(contract, 1, passage, zoom=1.0)
    plain = render_pdf_citation(contract, 1, "no such words appear anywhere here", zoom=1.0)
    assert marked.highlighted and not plain.highlighted
    assert marked.png != plain.png


@pytest.mark.parametrize(("page", "shown"), [(0, 1), (-3, 1), (2, 2), (99, 2)])
def test_out_of_range_page_is_clamped(contract: Path, page: int, shown: int) -> None:
    image = render_pdf_citation(contract, page, "words that appear on neither page")
    assert (image.page, image.highlighted) == (shown, False)


def test_passage_on_a_neighbouring_page_is_found(contract: Path) -> None:
    passage = "Termination does not affect obligations that arose before the notice was given."
    image = render_pdf_citation(contract, 1, passage)  # cited page 1, text is on page 2
    assert image.highlighted
    assert image.page == 2


def test_passage_straddling_a_page_break_stays_on_the_cited_page(contract: Path) -> None:
    passage = (
        "Payment is due within sixty days after the invoice date.\n\n"
        "Chapter Two\n\nEither party may terminate this agreement with ninety days written notice"
    )
    image = render_pdf_citation(contract, 1, passage)
    assert image.highlighted
    assert image.page == 1


def test_table_passage(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / "t.pdf", ["Monthly service fee for the premium support plan\n\n120 EUR"])
    passage = (
        "|Item|Price|\n|---|---|\n|Monthly service fee for the premium<br>support plan|120 EUR|"
    )
    assert render_pdf_citation(pdf, 1, passage).highlighted


def test_word_level_fallback_when_phrases_differ(tmp_path: Path) -> None:
    """The parser's word order can differ from the page's (right-to-left text, columns)."""
    pdf = _pdf(
        tmp_path / "w.pdf",
        ["Unrelated opening words.\n\nalpha bravo charlie delta echo foxtrot golf hotel india"],
    )
    shuffled = "india hotel golf foxtrot echo delta charlie bravo alpha"
    assert render_pdf_citation(pdf, 1, shuffled).highlighted
    # a few common words aren't enough
    assert not render_pdf_citation(pdf, 1, "alpha zulu yankee xray whiskey victor").highlighted


def test_encrypted_pdf_raises_a_readable_error(tmp_path: Path) -> None:
    doc: Any = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret", fontsize=12)
    path = tmp_path / "locked.pdf"
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
    doc.close()
    with pytest.raises(ValueError, match="password"):
        render_pdf_citation(path, 1, "secret")


def test_unreadable_pdf_raises_value_error(tmp_path: Path) -> None:
    junk = tmp_path / "junk.pdf"
    junk.write_bytes(b"this is not a pdf at all")
    with pytest.raises(ValueError, match=r"junk\.pdf"):
        render_pdf_citation(junk, 1, "anything")
    with pytest.raises(ValueError, match=r"missing\.pdf"):
        render_pdf_citation(tmp_path / "missing.pdf", 1, "anything")


def test_zoom_must_be_positive(contract: Path) -> None:
    with pytest.raises(ValueError, match="zoom"):
        render_pdf_citation(contract, 1, "Chapter One", zoom=0)


def test_highlighted_copy(contract: Path, tmp_path: Path) -> None:
    before = _digest(contract)
    out_dir = tmp_path / "out" / "nested"
    passage = "Either party may terminate this agreement with ninety days written notice"

    copy = highlighted_pdf_copy(contract, 2, passage, out_dir)

    assert copy == out_dir / "contract-highlight.pdf"
    assert _digest(contract) == before
    doc: Any = pymupdf.open(copy)
    try:
        assert [a.type[1] for a in doc[1].annots()] == ["Highlight"]
        assert list(doc[0].annots()) == []
        assert "OpenAction" in doc.xref_get_keys(doc.pdf_catalog())
    finally:
        doc.close()


# --- Hebrew (synthetic text only) ----------------------------------------------------

HEBREW = ["זהו משפט בדיקה ארוך שנכתב בעברית", "ושורה שנייה מתחתיו עם עוד מילים"]


def _hebrew_font() -> Any:
    try:
        return pymupdf.Font(script=5)  # MuPDF's built-in Noto Hebrew
    except Exception as e:
        pytest.skip(f"no Hebrew-capable font: {e}")


def test_hebrew_logical_order(tmp_path: Path) -> None:
    _hebrew_font()
    doc: Any = pymupdf.open()
    html = "".join(f'<p dir="rtl" style="font-size:14px">{line}</p>' for line in HEBREW)
    doc.new_page().insert_htmlbox(pymupdf.Rect(50, 50, 550, 780), html)
    path = tmp_path / "rtl.pdf"
    doc.save(path)
    doc.close()

    image = render_pdf_citation(path, 1, "\n".join(HEBREW))
    assert image.png.startswith(PNG)
    assert image.highlighted


def test_hebrew_stored_in_visual_order(tmp_path: Path) -> None:
    font = _hebrew_font()
    doc: Any = pymupdf.open()
    page = doc.new_page()
    page.insert_font(fontname="heb", fontbuffer=font.buffer)
    for i, line in enumerate(HEBREW):
        page.insert_text((72, 100 + 20 * i), line[::-1], fontname="heb", fontsize=12)
    path = tmp_path / "visual.pdf"
    doc.save(path)
    doc.close()

    image = render_pdf_citation(path, 1, "\n".join(HEBREW))
    assert image.png.startswith(PNG)
    assert image.highlighted


def test_hebrew_with_english_terms_does_not_raise(tmp_path: Path) -> None:
    _hebrew_font()
    doc: Any = pymupdf.open()
    html = '<p dir="rtl">אלגוריתם (Dynamic programming) פותר בעיות בעזרת טבלה של תתי בעיות</p>'
    doc.new_page().insert_htmlbox(pymupdf.Rect(50, 50, 550, 780), html)
    path = tmp_path / "mixed.pdf"
    doc.save(path)
    doc.close()

    passage = "אלגוריתם (Dynamic programming) פותר בעיות בעזרת טבלה של תתי בעיות"
    assert render_pdf_citation(path, 1, passage).png.startswith(PNG)


# --- passage pieces --------------------------------------------------------------------


def test_strip_markdown() -> None:
    text = "# **Title**\n\n- a _list_ item\n1. numbered `code`\n|a|b|\n|---|---|\n[link](x.md)"
    assert _strip_markdown(text).split("\n") == [
        "Title",
        "a list item",
        "numbered code",
        "a",
        "b",
        "link",
    ]
    assert _strip_markdown("snake_case and 2 * 3") == "snake_case and 2 * 3"


def test_pieces_are_sentences_and_bounded_windows() -> None:
    long = " ".join(f"w{i}" for i in range(30))
    pieces = _pieces(f"Too short. A sentence with exactly five words. {long}")
    assert pieces[0] == "A sentence with exactly five words."
    windows = pieces[1:]
    assert all(4 <= len(p.split()) <= 11 for p in windows)
    assert " ".join(windows) == long
    assert _pieces("Heading") == ["Heading"]  # a short passage is searched whole
    assert _pieces("") == []


# --- find_passage ----------------------------------------------------------------------

DOC = (
    "# Notes\n\nIntro line.\n\n"
    "The **renewal term** is one year,\r\nunless either party objects.\n\n"
    "| Fee | Amount |\n|---|---|\n| Setup<br>fee | 100 |\n"
)


def _span(document: str, passage: str) -> str | None:
    span = find_passage(document, passage)
    return None if span is None else document[span[0] : span[1]]


def test_find_passage_exact() -> None:
    assert _span(DOC, "Intro line.") == "Intro line."


def test_find_passage_whitespace_differences() -> None:
    assert _span(DOC, "is one year,  unless\neither party") == "is one year,\r\nunless either party"


def test_find_passage_markup_differences() -> None:
    assert _span(DOC, "The renewal term is one year") == "The **renewal term** is one year"
    assert _span(DOC, "Setup fee | 100") == "Setup<br>fee | 100"
    assert _span(DOC, "notes intro LINE") == "Notes\n\nIntro line"  # case and heading marks


def test_find_passage_not_found() -> None:
    assert find_passage(DOC, "a sentence that is not in the document") is None
    assert find_passage(DOC, "  ** ") is None
    assert find_passage("", "anything") is None


def test_find_passage_tolerates_an_edit_in_the_middle() -> None:
    head = "The licensee shall keep all confidential information secret and "
    tail = " and shall return it when the agreement ends for any reason."
    document = f"Preamble.\n\n{head}use it only for the project{tail}\n\nAfter."
    passage = f"{head}use it solely for the project{tail}"
    assert _span(document, passage) == f"{head}use it only for the project{tail}".strip()


def test_find_passage_offsets_survive_case_folding() -> None:
    document = "İstanbul office: the lease ends in May."
    assert _span(document, "the lease ends") == "the lease ends"
