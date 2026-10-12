from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pymupdf as _pymupdf
import pytest

from vaultmind.agent.baseline import NOT_FOUND, Source
from vaultmind.agent.types import Contradiction, UnsupportedClaim
from vaultmind.app import export
from vaultmind.app.export import answer_markdown, chat_markdown, markdown_pdf
from vaultmind.store.chats import Chat, ChatMessage

pymupdf: Any = _pymupdf  # partially typed upstream

T0 = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
HEBREW_Q = "מה תקופת ההודעה המוקדמת?"
HEBREW_A = "תקופת ההודעה היא שלושים יום [1]."


def _source(n: int, name: str | None, page: int | None, *section: str) -> Source:
    return Source(
        n=n,
        file_path=Path("vault") / name if name else None,
        page=page,
        section_path=section,
        text="synthetic passage",
        score=0.5,
    )


SOURCES = (
    _source(1, "lease.pdf", 3, "Terms", "Notice"),
    _source(2, "notes.md", None),
    _source(3, None, None),
)


def _q(text: str, seq: int = 0) -> ChatMessage:
    return ChatMessage("c", seq, "user", text, T0)


def _a(
    text: str,
    seq: int = 1,
    *,
    sources: tuple[Source, ...] = SOURCES,
    abstained: bool = False,
    contradictions: tuple[Contradiction, ...] = (),
    unsupported: tuple[UnsupportedClaim, ...] = (),
) -> ChatMessage:
    return ChatMessage(
        "c",
        seq,
        "assistant",
        text,
        T0,
        run_id=None,
        sources=sources,
        abstained=abstained,
        contradictions=contradictions,
        unsupported=unsupported,
    )


@pytest.fixture
def no_system_font(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    export._font_files.cache_clear()
    monkeypatch.setattr(export, "_FONT_CANDIDATES", ())
    yield
    export._font_files.cache_clear()


def _pdf_text(data: bytes) -> str:
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        # Some fonts map the space glyph to U+00A0 on extraction.
        return " ".join(page.get_text() for page in doc).replace("\xa0", " ")
    finally:
        doc.close()


def test_answer_lists_only_cited_sources() -> None:
    md = answer_markdown(
        _q("What is the notice period?"), _a("Thirty days [1], see also [3] and [9].")
    )
    assert md == (
        "## What is the notice period?\n"
        "\n"
        "Thirty days [1], see also [3] and [9].\n"
        "\n"
        "**Sources**\n"
        "\n"
        "- [1] lease.pdf, p. 3, Terms > Notice\n"
        "- [3] unknown file\n"
    )


def test_answer_without_citations_lists_all_sources() -> None:
    md = answer_markdown(_q("Q?"), _a("An answer.\n\n- point one\n- point two"))
    assert md == (
        "## Q?\n\n"
        "An answer.\n\n- point one\n- point two\n\n"
        "**Sources**\n\n"
        "- [1] lease.pdf, p. 3, Terms > Notice\n"
        "- [2] notes.md\n"
        "- [3] unknown file\n"
    )


def test_abstained_lists_closest_matches() -> None:
    md = answer_markdown(_q("Who won?"), _a(NOT_FOUND, sources=SOURCES[:2], abstained=True))
    assert md == (
        "## Who won?\n\n"
        f"{NOT_FOUND}\n\n"
        "**Closest matches**\n\n"
        "- [1] lease.pdf, p. 3, Terms > Notice\n"
        "- [2] notes.md\n"
    )
    bare = answer_markdown(_q("Who won?"), _a(NOT_FOUND, sources=(), abstained=True))
    assert bare == f"## Who won?\n\n{NOT_FOUND}\n"


def test_contradictions_and_unsupported_notes() -> None:
    answer = _a(
        "The sources disagree [1][2].",
        contradictions=(Contradiction("notice is 30 days", 1, "notice is\n60 days", 2),),
        unsupported=(
            UnsupportedClaim("A fee applies.", (2,), "no source mentions a fee"),
            UnsupportedClaim("Uncited claim.", (), ""),
        ),
    )
    assert answer_markdown(_q("  Notice\nperiod? "), answer) == (
        "## Notice period?\n\n"
        "The sources disagree [1][2].\n\n"
        "**Sources**\n\n"
        "- [1] lease.pdf, p. 3, Terms > Notice\n"
        "- [2] notes.md\n\n"
        '**Note:** sources [1] and [2] disagree: "notice is 30 days" vs. "notice is 60 days"\n\n'
        '**Note:** possibly unsupported by its sources: "A fee applies." '
        "(cited [2]; no source mentions a fee)\n\n"
        '**Note:** possibly unsupported by its sources: "Uncited claim." (cited no source)\n'
    )


def test_chat_markdown_pairs_exchanges_and_keeps_trailing_question() -> None:
    chat = Chat("c", "Lease questions", T0, T0)
    messages = [
        _q("What is the notice period?", 0),
        _a("Thirty days [1].", 1),
        _q("Who won the match?", 2),
        _a(NOT_FOUND, 3, sources=SOURCES[1:2], abstained=True),
        _q("Unanswered question?", 4),
    ]
    md = chat_markdown(chat, messages, now=datetime(2026, 10, 12, 14, 3))
    assert md == (
        "# Lease questions\n\n"
        "Exported from VaultMind on 2026-10-12 14:03\n\n"
        "## What is the notice period?\n\n"
        "Thirty days [1].\n\n"
        "**Sources**\n\n"
        "- [1] lease.pdf, p. 3, Terms > Notice\n\n"
        "## Who won the match?\n\n"
        f"{NOT_FOUND}\n\n"
        "**Closest matches**\n\n"
        "- [2] notes.md\n\n"
        "## Unanswered question?\n"
    )


def test_chat_markdown_default_time_and_empty_chat() -> None:
    md = chat_markdown(Chat("c", "Empty", T0, T0), [])
    assert md.startswith("# Empty\n\nExported from VaultMind on ")
    assert md.endswith("\n") and md.count("\n\n") == 1


def test_html_direction_and_escaping() -> None:
    md = answer_markdown(
        _q(HEBREW_Q),
        _a(
            f"{HEBREW_A}\n\n- **פריט** & <one>\n- item two",
            sources=(_source(1, "חוזה.pdf", 2),),
        ),
    )
    html = export._to_html(md)
    assert f'<h2 dir="rtl">{HEBREW_Q}</h2>' in html
    assert '<p dir="rtl">תקופת ההודעה' in html
    assert '<p dir="rtl" class="item-rtl">• <b>פריט</b> &amp; &lt;one&gt;</p>' in html
    assert '<p class="item">• item two</p>' in html
    assert "<p><b>Sources</b></p>" in html
    # A source line stays left-to-right even with a Hebrew file name.
    assert '<p class="item">• [1] חוזה.pdf, p. 2</p>' in html


def test_pdf_english() -> None:
    chat = Chat("c", "Lease questions", T0, T0)
    md = chat_markdown(chat, [_q("What is the notice period?"), _a("Thirty days [1].")])
    data = markdown_pdf(md)
    assert data.startswith(b"%PDF")
    text = _pdf_text(data)
    for phrase in ("Lease questions", "What is the notice period?", "Thirty days [1].", "p. 3"):
        assert phrase in text
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        assert doc.metadata is not None and doc.metadata["title"] == "Lease questions"
    finally:
        doc.close()


def test_pdf_hebrew_with_system_font() -> None:
    export._font_files.cache_clear()
    if export._font_files() is None:
        pytest.skip("no Hebrew-capable system font found (Arial, DejaVu Sans, ...)")
    md = answer_markdown(_q(HEBREW_Q), _a(HEBREW_A, sources=(_source(1, "lease.pdf", 3),)))
    text = _pdf_text(markdown_pdf(md))
    for word in ("תקופת", "ההודעה", "המוקדמת", "שלושים", "lease.pdf"):
        assert word in text


def test_pdf_without_system_font(no_system_font: None) -> None:
    md = answer_markdown(_q(HEBREW_Q), _a(f"Thirty days. {HEBREW_A}"))
    text = _pdf_text(markdown_pdf(md))
    assert "Thirty days." in text
    # MuPDF's bundled Noto Serif Hebrew covers the Hebrew glyphs.
    assert "שלושים" in text


def test_pdf_long_chat_spans_pages() -> None:
    chat = Chat("c", "Long", T0, T0)
    messages: list[ChatMessage] = []
    for i in range(40):
        messages += [_q(f"Question number {i}?", 2 * i), _a(f"Answer number {i} [1].", 2 * i + 1)]
    data = markdown_pdf(chat_markdown(chat, messages))
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        assert doc.page_count > 1
    finally:
        doc.close()
    assert "Answer number 39 [1]." in _pdf_text(data)
