import itertools
import re
from pathlib import Path

import pytest

from vaultmind.core.config import ChunkingConfig
from vaultmind.core.models import Block, ChildChunk, ParentSection, ParsedDocument
from vaultmind.ingest.chunker import chunk_document, estimate_tokens

FILE_ID = "f00d"
SMALL = ChunkingConfig(child_tokens=40, parent_tokens=200, overlap=0.2)
CONFIGS = [
    SMALL,
    ChunkingConfig(child_tokens=25, parent_tokens=100, overlap=0.5),
    ChunkingConfig(child_tokens=60, parent_tokens=150, overlap=0.0),
    ChunkingConfig(),
]

# (page, section_path, sentence count) for a multi-section, multi-page document
LAYOUT: list[tuple[int, tuple[str, ...], int]] = [
    (1, ("Guide",), 3),
    (1, ("Guide", "Setup"), 12),
    (2, ("Guide", "Setup"), 9),
    (2, ("Guide", "Usage"), 30),
    (3, ("Guide", "Usage"), 4),
    (3, ("Guide", "Usage", "Advanced"), 20),
    (4, ("Appendix",), 6),
]
_TAG = re.compile(r"\[p(\d+) ([^\]]*)\]")


def _sentence(page: int, path: tuple[str, ...], k: int) -> str:
    """Each sentence starts with a tag naming where it came from."""
    words = " ".join(f"word{(k * 7 + w) % 13}" for w in range(k % 5 + 3))
    return f"[p{page} {'/'.join(path)}] Sentence {k} has {words}."


def _layout_doc() -> ParsedDocument:
    blocks = []
    for page, path, n in LAYOUT:
        sentences = [_sentence(page, path, k) for k in range(n)]
        paragraphs = [" ".join(sentences[i : i + 4]) for i in range(0, n, 4)]
        blocks.append(Block("\n\n".join(paragraphs), page, path))
    return ParsedDocument(path=Path("guide.pdf"), title="Guide Book", blocks=blocks)


def _awkward_doc() -> ParsedDocument:
    """Blocks larger than a parent, with no paragraph or sentence boundaries."""
    words = " ".join(f"tok{i}" for i in range(900))
    run_on = "".join(str(i) for i in range(1100))  # no whitespace, but not periodic
    many_paras = "\n\n".join(f"Para {i} short." for i in range(150))
    return ParsedDocument(
        path=Path("a.md"),
        title="Awkward",
        blocks=[
            Block(words, None, ("Words",)),
            Block(run_on, None, ("Run on",)),
            Block("   \n ", None, ("Blank",)),
            Block(many_paras, None, ()),
        ],
    )


DOCS = [_layout_doc(), _awkward_doc()]


def _child_spans(parent: ParentSection, children: list[ChildChunk]) -> list[tuple[int, int]]:
    """Locate each child (a slice of its parent) in the parent's text, in order."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for child in children:
        start = parent.text.find(child.text, pos)
        assert start >= 0, f"{child.id} is not a slice of its parent"
        spans.append((start, start + len(child.text)))
        pos = start + 1
    return spans


def _by_parent(children: list[ChildChunk]) -> dict[str, list[ChildChunk]]:
    out: dict[str, list[ChildChunk]] = {}
    for c in children:
        out.setdefault(c.parent_id, []).append(c)
    return out


def test_estimate_tokens() -> None:
    assert [estimate_tokens(t) for t in ["", "a", "abcd", "abcde", "x" * 400]] == [0, 1, 1, 2, 100]


def test_empty_document() -> None:
    empty = ParsedDocument(path=Path("e.txt"), title="e", blocks=[])
    assert chunk_document(empty, FILE_ID, SMALL) == ([], [])
    blank = ParsedDocument(path=Path("e.txt"), title="e", blocks=[Block(" \n", None, ())])
    assert chunk_document(blank, FILE_ID, SMALL) == ([], [])


@pytest.mark.parametrize("doc", DOCS, ids=["layout", "awkward"])
@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: f"{c.child_tokens}-{c.parent_tokens}")
def test_size_limits(doc: ParsedDocument, cfg: ChunkingConfig) -> None:
    parents, children = chunk_document(doc, FILE_ID, cfg)
    assert parents and children
    assert max(estimate_tokens(p.text) for p in parents) <= cfg.parent_tokens * 1.1
    assert max(estimate_tokens(c.text) for c in children) <= cfg.child_tokens * 1.1


@pytest.mark.parametrize("doc", DOCS, ids=["layout", "awkward"])
@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: f"{c.child_tokens}-{c.parent_tokens}")
def test_children_cover_parents(doc: ParsedDocument, cfg: ChunkingConfig) -> None:
    parents, children = chunk_document(doc, FILE_ID, cfg)
    groups = _by_parent(children)
    assert set(groups) == {p.id for p in parents}
    for parent in parents:
        covered = [False] * len(parent.text)
        for start, end in _child_spans(parent, groups[parent.id]):
            covered[start:end] = [True] * (end - start)
        gaps = [ch for ch, ok in zip(parent.text, covered, strict=True) if not ok]
        assert all(ch.isspace() for ch in gaps), f"{parent.id} has uncovered text"


@pytest.mark.parametrize("doc", DOCS, ids=["layout", "awkward"])
def test_all_block_text_is_kept(doc: ParsedDocument) -> None:
    parents, _ = chunk_document(doc, FILE_ID, SMALL)
    original = "".join("".join(b.text.split()) for b in doc.blocks)
    assert "".join("".join(p.text.split()) for p in parents) == original


def test_ids_unique_and_well_formed() -> None:
    parents, children = chunk_document(_layout_doc(), FILE_ID, SMALL)
    assert [p.id for p in parents] == [f"{FILE_ID}:p{i}" for i in range(len(parents))]
    ids = [c.id for c in children]
    assert len(set(ids)) == len(ids)
    for parent_id, group in _by_parent(children).items():
        assert [c.id for c in group] == [f"{parent_id}:c{j}" for j in range(len(group))]
    assert all(c.file_id == FILE_ID and p.file_id == FILE_ID for c in children for p in parents)


@pytest.mark.parametrize("doc", DOCS, ids=["layout", "awkward"])
def test_deterministic(doc: ParsedDocument) -> None:
    assert chunk_document(doc, FILE_ID, SMALL) == chunk_document(doc, FILE_ID, SMALL)


@pytest.mark.parametrize("cfg", CONFIGS[:2], ids=["small", "half-overlap"])
def test_pages_and_sections(cfg: ChunkingConfig) -> None:
    doc = _layout_doc()
    parents, children = chunk_document(doc, FILE_ID, cfg)
    assert len({p.section_path for p in parents}) > 2
    assert len({c.page for c in children}) == 4

    for child in children:
        m = _TAG.match(child.text)
        if m:  # the child starts with a sentence: its tag says where it came from
            assert child.page == int(m.group(1))
            assert "/".join(child.section_path) == m.group(2)
        assert child.embed_text == " > ".join(("Guide Book", *child.section_path)) + (
            "\n\n" + child.text
        )

    for parent in parents:
        tags = [(int(p), s) for p, s in _TAG.findall(parent.text)]
        first = _TAG.match(parent.text)
        assert first is not None
        assert "/".join(parent.section_path) == first.group(2)
        assert parent.page_start == min(p for p, _ in tags)
        assert parent.page_end == max(p for p, _ in tags)


def test_child_page_is_where_text_starts() -> None:
    """A child starting at the end of page 1 keeps page 1 even if it runs onto page 2."""
    doc = _layout_doc()
    parents, children = chunk_document(doc, FILE_ID, ChunkingConfig(child_tokens=60))
    spanning = [c for c in children if "[p1" in c.text and "[p2" in c.text]
    assert spanning
    assert all(c.page == 1 for c in spanning if c.text.startswith("[p1"))
    assert parents[0].page_start == 1


def test_new_parent_at_heading_change_when_half_full() -> None:
    a = Block("Alpha sentence. " * 30, 1, ("A",))  # ~120 tokens: over half of 200
    b = Block("Beta sentence. " * 5, 1, ("B",))  # would fit in the same parent
    tiny = Block("Gamma.", 1, ("C",))
    doc = ParsedDocument(path=Path("x.md"), title="X", blocks=[tiny, a, b])
    parents, _ = chunk_document(doc, FILE_ID, SMALL)
    # tiny + a share a parent (tiny was not half full); b starts a new one
    assert [p.section_path for p in parents] == [("C",), ("B",)]
    assert "Alpha" in parents[0].text and "Beta" not in parents[0].text


def test_overlap_between_consecutive_children() -> None:
    parents, children = chunk_document(_layout_doc(), FILE_ID, SMALL)
    groups = _by_parent(children)
    pairs = 0
    for parent in parents:
        spans = _child_spans(parent, groups[parent.id])
        for (_, prev_end), (next_start, next_end) in itertools.pairwise(spans):
            assert next_start < prev_end < next_end  # shares text, then moves on
            pairs += 1
    assert pairs > 5


def test_no_overlap_when_disabled() -> None:
    parents, children = chunk_document(
        _layout_doc(), FILE_ID, ChunkingConfig(child_tokens=40, parent_tokens=200, overlap=0)
    )
    groups = _by_parent(children)
    for parent in parents:
        spans = _child_spans(parent, groups[parent.id])
        assert all(a[1] <= b[0] for a, b in itertools.pairwise(spans))


def test_children_prefer_sentence_boundaries() -> None:
    cfg = ChunkingConfig(child_tokens=60, parent_tokens=300, overlap=0.2)
    _, children = chunk_document(_layout_doc(), FILE_ID, cfg)
    assert all(c.text.endswith(".") for c in children)
    assert all(c.text.startswith("[p") for c in children)
