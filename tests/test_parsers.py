from pathlib import Path
from typing import Any

import pymupdf as _pymupdf
import pytest

from vaultmind.core.models import Block
from vaultmind.ingest.parsers import ParseError, UnsupportedFileError, parse

pymupdf: Any = _pymupdf  # partially typed upstream

HEADING_SIZES = {1: 24, 2: 18}
BODY = "This is ordinary body text that fills the line with words."


def _make_pdf(path: Path, pages: list[list[tuple[int, str]]], *, title: str = "") -> Path:
    """Each page is a list of (level, text): level 0 is body text, 1/2 are headings."""
    doc: Any = pymupdf.open()
    for lines in pages:
        page = doc.new_page()
        y = 72.0
        for level, text in lines:
            size = HEADING_SIZES.get(level, 11)
            if level:
                y += 12
            page.insert_text((72, y), text, fontsize=size)
            y += size + 6
    if title:
        doc.set_metadata({"title": title})
    doc.save(path)
    doc.close()
    return path


def _body(tag: str, n: int = 4) -> list[tuple[int, str]]:
    return [(0, f"{tag} line {i}. {BODY}") for i in range(n)]


def _find(blocks: list[Block], needle: str) -> Block:
    return next(b for b in blocks if needle in b.text)


def _lines(blocks: list[Block]) -> list[str]:
    return [line for b in blocks for line in b.text.splitlines() if line.strip()]


# --- PDF -----------------------------------------------------------------------------


def test_pdf_heading_continues_across_pages(tmp_path: Path) -> None:
    pdf = _make_pdf(
        tmp_path / "guide.pdf",
        [
            [(1, "User Guide"), (2, "Setup"), *_body("setup-one")],
            [*_body("setup-two"), (2, "Usage"), *_body("usage")],
        ],
        title="Metadata Title",
    )
    doc = parse(pdf)

    assert doc.title == "User Guide"  # first H1 wins over metadata
    assert doc.path == pdf
    first = _find(doc.blocks, "setup-one line 0")
    assert (first.page, first.section_path) == (1, ("User Guide", "Setup"))
    continued = _find(doc.blocks, "setup-two line 0")
    assert (continued.page, continued.section_path) == (2, ("User Guide", "Setup"))
    usage = _find(doc.blocks, "usage line 0")
    assert (usage.page, usage.section_path) == (2, ("User Guide", "Usage"))
    # one block per heading-delimited span per page
    assert "setup-two" not in usage.text
    assert "usage" not in continued.text
    assert all(b.text.strip() == b.text and b.text for b in doc.blocks)


def test_pdf_title_falls_back_to_metadata_then_stem(tmp_path: Path) -> None:
    with_meta = _make_pdf(tmp_path / "a.pdf", [_body("plain")], title="  Annual Report ")
    assert parse(with_meta).title == "Annual Report"
    without = _make_pdf(tmp_path / "quarterly-notes.pdf", [_body("plain")])
    doc = parse(without)
    assert doc.title == "quarterly-notes"
    assert doc.blocks[0].section_path == ()
    assert doc.blocks[0].page == 1


def test_pdf_title_ignores_h1_after_the_first_page(tmp_path: Path) -> None:
    pdf = _make_pdf(
        tmp_path / "formulas.pdf",
        [[(2, "Data formulas"), *_body("one")], [(1, "Learn more"), *_body("two")]],
        title="Formula Reference",
    )
    assert parse(pdf).title == "Formula Reference"


def test_pdf_uppercase_extension(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "UPPER.PDF", [_body("upper")])
    assert "upper line 0" in parse(pdf).blocks[0].text


def test_encrypted_pdf_raises(tmp_path: Path) -> None:
    doc: Any = pymupdf.open()
    doc.new_page().insert_text((72, 72), "top secret", fontsize=12)
    path = tmp_path / "locked.pdf"
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="user", owner_pw="owner")
    doc.close()
    with pytest.raises(ParseError, match="password"):
        parse(path)


def test_pdf_without_text_raises(tmp_path: Path) -> None:
    doc: Any = pymupdf.open()
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(72, 72, 300, 300), color=(0, 0, 0), fill=(0.5, 0.5, 0.5))
    path = tmp_path / "scan.pdf"
    doc.save(path)
    doc.close()
    with pytest.raises(ParseError, match="OCR"):
        parse(path)


@pytest.mark.parametrize("content", [b"definitely not a pdf", b""])
def test_corrupt_pdf_raises(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(content)
    with pytest.raises(ParseError):
        parse(path)


def _hidden_text_pdf(path: Path, lines: list[str], **style: Any) -> Path:
    """A page whose only text is invisible, like an OCR or handwriting-recognition layer."""
    doc: Any = pymupdf.open()
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(36, 36, 560, 800), color=(0, 0, 0), fill=(0.9, 0.9, 0.9))
    for i, line in enumerate(lines):
        page.insert_text((72, 72 + 16 * i), line, fontsize=11, **style)
    doc.save(path)
    doc.close()
    return path


@pytest.mark.parametrize("style", [{"render_mode": 3}, {"fill_opacity": 0}])
def test_garbled_hidden_text_layer_raises(tmp_path: Path, style: dict[str, Any]) -> None:
    junk = [" ".join(f"q{i}x{j}{i * j}zk m{j}v{i}0 rr{i}{j}t" for j in range(6)) for i in range(30)]
    path = _hidden_text_pdf(tmp_path / "notes.pdf", junk, **style)
    with pytest.raises(ParseError, match="garbled"):
        parse(path)


def test_clean_hidden_text_layer_is_not_rejected(tmp_path: Path) -> None:
    clean = [f"Line {i} of a scanned page with a clean recognized text layer." for i in range(30)]
    path = _hidden_text_pdf(tmp_path / "scan.pdf", clean, render_mode=3)
    try:
        parse(path)
    except ParseError as e:  # pymupdf4llm may drop invisible text; it must not be "garbled"
        assert "garbled" not in str(e)


# --- right-to-left PDF ---------------------------------------------------------------


def _html_pdf(path: Path, html: str) -> Path:
    """MuPDF lays the HTML out with the Unicode bidi algorithm and stores logical order,
    like word processors and slide tools do."""
    doc: Any = pymupdf.open()
    doc.new_page().insert_htmlbox(pymupdf.Rect(50, 50, 550, 780), html)
    doc.save(path)
    doc.close()
    return path


def _rtl(text: str, size: int = 12, tag: str = "p") -> str:
    return f'<{tag} dir="rtl" style="font-size:{size}px">{text}</{tag}>'


def test_hebrew_pdf_reads_in_logical_order(tmp_path: Path) -> None:
    pdf = _html_pdf(
        tmp_path / "lecture.pdf",
        _rtl("מבני נתונים", 24, "h1")
        + _rtl("שלום עולם (בדיקה) עם Knapsack ב-2026 השנה.")
        + _rtl("מועד אחרון להגשה: 15.9.2026 בשעה 23:59.")
        + _rtl("הנוסחה x + y = 5 נכונה, וגם f(n) = 2n.")
        + '<p style="font-size:12px">An English line (with brackets) stays as it is.</p>',
    )
    doc = parse(pdf)

    assert doc.title == "מבני נתונים"
    text = "\n".join(b.text for b in doc.blocks)
    assert "שלום עולם (בדיקה) עם Knapsack ב-2026 השנה." in text  # words, brackets, numbers
    assert "מועד אחרון להגשה: 15.9.2026 בשעה 23:59." in text  # a date is not reversed
    assert "הנוסחה x + y = 5 נכונה, וגם f(n) = 2n." in text  # formulas keep their order
    assert "An English line (with brackets) stays as it is." in text
    body = _find(doc.blocks, "שלום")
    assert (body.page, body.section_path) == (1, ("מבני נתונים",))


def test_hebrew_stored_in_visual_order(tmp_path: Path) -> None:
    """Some producers write RTL text reversed, left to right; position decides the order."""
    font: Any = pymupdf.Font(script=5)  # MuPDF's built-in Noto Hebrew
    doc: Any = pymupdf.open()
    page = doc.new_page()
    page.insert_font(fontname="heb", fontbuffer=font.buffer)
    logical = ["זהו משפט שנשמר בסדר חזותי", "ושורה שנייה מתחתיו"]
    for i, line in enumerate(logical):
        page.insert_text((72, 100 + 20 * i), line[::-1], fontname="heb", fontsize=12)
    path = tmp_path / "visual.pdf"
    doc.save(path)
    doc.close()

    assert _lines(parse(path).blocks) == logical


def test_hebrew_list_markers_and_brackets(tmp_path: Path) -> None:
    pdf = _html_pdf(
        tmp_path / "list.pdf",
        _rtl("1. פריט ראשון (חשוב)") + _rtl("2. פריט שני [ב] {ג}") + _rtl("הערה (Gen AI) בסוף"),
    )
    assert _lines(parse(pdf).blocks) == [
        "1. פריט ראשון (חשוב)",
        "2. פריט שני [ב] {ג}",
        "הערה (Gen AI) בסוף",
    ]


def test_hebrew_table_keeps_rows_and_rtl_column_order(tmp_path: Path) -> None:
    doc: Any = pymupdf.open()
    page = doc.new_page()
    rows = [("נושא", "שבוע"), ("מיון מהיר", "1"), ("עץ פורש מינימלי", "2")]
    x0, x1, x2, top, height = 100.0, 300.0, 500.0, 100.0, 30.0
    for r, (right, left) in enumerate(rows):
        y = top + r * height
        cells = ((x1, y, x2, y + height, right), (x0, y, x1, y + height, left))
        for cx0, cy0, cx1, cy1, text in cells:
            page.draw_rect(pymupdf.Rect(cx0, cy0, cx1, cy1), color=(0, 0, 0), width=1)
            box = pymupdf.Rect(cx0 + 4, cy0 + 4, cx1 - 4, cy1 - 4)
            page.insert_htmlbox(box, _rtl(text, 11))
    page.insert_htmlbox(pymupdf.Rect(100, 300, 500, 400), _rtl("טקסט אחרי הטבלה."))
    path = tmp_path / "table.pdf"
    doc.save(path)
    doc.close()

    text = "\n".join(b.text for b in parse(path).blocks)
    assert "| נושא | שבוע |" in text  # the right-hand column comes first
    assert "| מיון מהיר | 1 |" in text
    assert "| עץ פורש מינימלי | 2 |" in text
    assert text.index("עץ פורש") < text.index("טקסט אחרי הטבלה.")


def test_overlapping_text_boxes_are_not_interleaved(tmp_path: Path) -> None:
    """A callout drawn over other text at almost the same height stays a separate row."""
    doc: Any = pymupdf.open()
    page = doc.new_page()
    page.insert_htmlbox(pymupdf.Rect(100, 100, 500, 130), _rtl("שורה ראשית בגוף השקף"))
    page.insert_htmlbox(pymupdf.Rect(100, 102, 500, 132), _rtl("הערה צדדית מעל השורה"))
    path = tmp_path / "callout.pdf"
    doc.save(path)
    doc.close()

    lines = _lines(parse(path).blocks)
    assert "שורה ראשית בגוף השקף" in lines
    assert "הערה צדדית מעל השורה" in lines


@pytest.mark.parametrize(
    ("visual", "rtl", "logical"),
    [
        ("םולש", True, "שלום"),
        ("(הקידב)", True, "(בדיקה)"),  # the shown glyphs are stored
        (")הקידב(", True, "(בדיקה)"),  # the logical characters are stored, the font mirrors
        ("8 = 3 + 5 :ךרע", True, "ערך: 5 + 3 = 8"),  # numbers alone follow the RTL order
        ("x + y = 5 :ךרע", True, "ערך: x + y = 5"),  # a formula with letters stays LTR
        ("ab (םולש) cd", False, "ab (שלום) cd"),
        ("5.2 :הסרג", True, "גרסה: 5.2"),
        (".2 :ףיעס", True, "סעיף: 2."),
        ("ףיעס .2", True, "2. סעיף"),  # a list marker starts the line
        ("80% :יחכונ", True, "נוכחי: 80%"),
    ],
)
def test_visual_to_logical(visual: str, rtl: bool, logical: str) -> None:
    from vaultmind.ingest.parsers import _visual_to_logical

    assert _visual_to_logical(list(visual), rtl) == logical


@pytest.mark.parametrize(
    ("code", "font", "expected"),
    [
        (0xF051, "SymbolMT", "Θ"),
        (0xF061, "SymbolMT", "α"),  # noqa: RUF001
        (0xF0AC, "SymbolMT", "←"),
        (0xF0CE, "SymbolMT", "∈"),
        (0xF028, "SymbolMT", "("),
        (0xF0DF, "Wingdings-Regular", "←"),
        (0xF0FE, "Wingdings-Regular", ""),  # an unknown icon is dropped
        (0xE123, "SomeIconFont", ""),
    ],
)
def test_symbol_font_private_use_characters(code: int, font: str, expected: str) -> None:
    from vaultmind.ingest.parsers import _private_use

    assert _private_use(chr(code), font) == expected


# --- Markdown ------------------------------------------------------------------------


def test_markdown_headings_and_code_fence(tmp_path: Path) -> None:
    md = tmp_path / "notes.md"
    md.write_text(
        "Preamble before any heading.\n\n"
        "# Project Notes #\n\nIntro text.\n\n"
        "## Install\n\nRun this:\n\n"
        "```bash\n# not a heading\npip install thing\n```\n\n"
        "~~~\n## also not a heading\n~~~\n\n"
        "### Details\n\nDeep text.\n\n"
        "## Usage\n\nUse it.\n\n"
        "#hashtag is not a heading\n",
        encoding="utf-8",
    )
    doc = parse(md)

    assert doc.title == "Project Notes"
    paths = [(b.section_path, b.page) for b in doc.blocks]
    assert paths == [
        ((), None),
        (("Project Notes",), None),
        (("Project Notes", "Install"), None),
        (("Project Notes", "Install", "Details"), None),
        (("Project Notes", "Usage"), None),
    ]
    install = doc.blocks[2]
    assert install.text.startswith("## Install")
    assert "# not a heading" in install.text
    assert "## also not a heading" in install.text
    assert doc.blocks[4].text.endswith("#hashtag is not a heading")


def test_markdown_without_h1_uses_stem(tmp_path: Path) -> None:
    md = tmp_path / "readme.markdown"
    md.write_text("## Only H2\n\ntext\n", encoding="utf-8")
    doc = parse(md)
    assert doc.title == "readme"
    assert doc.blocks[0].section_path == ("Only H2",)


def test_markdown_bom_and_crlf(tmp_path: Path) -> None:
    md = tmp_path / "bom.md"
    md.write_bytes("﻿# Título\r\n\r\nCafé text.\r\n".encode())
    doc = parse(md)
    assert doc.title == "Título"
    assert doc.blocks[0].text == "# Título\n\nCafé text."


# --- plain text ----------------------------------------------------------------------


def test_text_paragraphs(tmp_path: Path) -> None:
    txt = tmp_path / "memo.txt"
    txt.write_text(
        "\n\nFirst paragraph\nstill first.\n\n  \t\nSecond paragraph.\n\n\n\n# Third\n\n   \n",
        encoding="utf-8",
    )
    doc = parse(txt)
    assert doc.title == "memo"
    assert doc.blocks == [
        Block("First paragraph\nstill first.", None, ()),
        Block("Second paragraph.", None, ()),
        Block("# Third", None, ()),
    ]


def test_text_bom_is_stripped(tmp_path: Path) -> None:
    txt = tmp_path / "bom.txt"
    txt.write_bytes(b"\xef\xbb\xbfhello")
    assert parse(txt).blocks[0].text == "hello"


def test_invalid_utf8_raises(tmp_path: Path) -> None:
    txt = tmp_path / "latin1.txt"
    txt.write_bytes("café".encode("latin-1"))
    with pytest.raises(ParseError, match="UTF-8"):
        parse(txt)


@pytest.mark.parametrize("name", ["empty.txt", "blank.md"])
def test_whitespace_only_file_raises(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_text("  \n\n\t\n", encoding="utf-8")
    with pytest.raises(ParseError):
        parse(path)


@pytest.mark.parametrize("name", ["report.docx", "noext", "image.png"])
def test_unsupported_extension(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_text("x", encoding="utf-8")
    with pytest.raises(UnsupportedFileError):
        parse(path)
