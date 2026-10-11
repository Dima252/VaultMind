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
