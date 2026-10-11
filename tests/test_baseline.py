from collections.abc import Iterator
from pathlib import Path

import pytest

from vaultmind.agent.baseline import (
    NOT_FOUND,
    Source,
    answer,
    build_messages,
    cited_sources,
    is_abstention,
    retrieve,
    stream_answer,
)
from vaultmind.core.config import Config
from vaultmind.core.models import ChildChunk, FileRecord, ParentSection
from vaultmind.store.index import ChunkIndex
from vaultmind.store.registry import Registry

from .fakes import FakeEmbedder, FakeLLM

FILE_ID = "0123456789abcdef"
CFG = Config()


def _child(parent: ParentSection, j: int, text: str, page: int) -> ChildChunk:
    return ChildChunk(
        id=f"{parent.id}:c{j}",
        parent_id=parent.id,
        file_id=FILE_ID,
        text=text,
        embed_text=f"Manual > Setup\n\n{text}",
        page=page,
        section_path=parent.section_path,
    )


PARENTS = [
    ParentSection(f"{FILE_ID}:p0", FILE_ID, "Setup section text.", ("Setup",), 2, 3),
    ParentSection(f"{FILE_ID}:p1", FILE_ID, "Usage section text.", ("Usage",), 4, 4),
]
CHILDREN = [
    _child(PARENTS[0], 0, "Install with uv.", 2),
    _child(PARENTS[0], 1, "Then run doctor.", 3),
    _child(PARENTS[1], 0, "Ask a question.", 4),
]


@pytest.fixture
def store(tmp_path: Path) -> Iterator[tuple[ChunkIndex, Registry]]:
    embedder = FakeEmbedder()
    index = ChunkIndex(tmp_path / "index", embedder_name=embedder.model_name)
    registry = Registry(tmp_path / "vaultmind.sqlite")
    index.add(PARENTS, CHILDREN, embedder.embed([c.embed_text for c in CHILDREN]))
    registry.upsert(FileRecord(FILE_ID, tmp_path / "manual.pdf", "sha", "ready"))
    yield index, registry
    registry.close()
    index.close()


def test_retrieve_collapses_children_to_parents(store: tuple[ChunkIndex, Registry]) -> None:
    index, registry = store
    question = CHILDREN[1].embed_text  # the fake embedder matches identical text exactly
    sources = retrieve(question, CFG, embedder=FakeEmbedder(), index=index, registry=registry)

    assert [s.n for s in sources] == [1, 2]
    top = sources[0]
    assert top.text == "Setup section text."  # the LLM reads the parent section
    assert top.page == 3  # cited at the best-matching child's page
    assert top.file_path is not None and top.file_path.name == "manual.pdf"
    assert top.label == "manual.pdf, p. 3, Setup"
    assert top.score == pytest.approx(1.0)


def test_answer_maps_citations_to_sources(store: tuple[ChunkIndex, Registry]) -> None:
    index, registry = store
    llm = FakeLLM(["Run doctor after installing [1]. Also see [9]."])
    result = answer(
        CHILDREN[1].embed_text,
        CFG,
        llm=llm,
        embedder=FakeEmbedder(),
        index=index,
        registry=registry,
    )
    assert [s.n for s in result.cited] == [1]
    assert not result.abstained
    user_prompt = llm.calls[0][1]["content"]
    assert "[1] (manual.pdf, p. 3, Setup)\nSetup section text." in user_prompt


def test_empty_index_abstains_without_calling_llm(tmp_path: Path) -> None:
    index = ChunkIndex(tmp_path / "index", embedder_name=FakeEmbedder.model_name)
    registry = Registry(tmp_path / "r.sqlite")
    llm = FakeLLM(["should not be used"])
    try:
        result = answer("q", CFG, llm=llm, embedder=FakeEmbedder(), index=index, registry=registry)
        sources, tokens = stream_answer(
            "q", CFG, llm=llm, embedder=FakeEmbedder(), index=index, registry=registry
        )
    finally:
        registry.close()
    assert result.text == NOT_FOUND and result.abstained and result.sources == []
    assert sources == [] and "".join(tokens) == NOT_FOUND
    assert llm.calls == []


def test_build_messages_numbers_sources() -> None:
    sources = [Source(1, Path("a.md"), None, (), "Alpha.", 0.9)]
    system, user = build_messages("What is alpha?", sources)
    assert NOT_FOUND in system["content"]
    assert user["content"].endswith("[1] (a.md)\nAlpha.\n\nQuestion: What is alpha?")


def test_is_abstention_and_cited_sources() -> None:
    assert is_abstention(NOT_FOUND)
    assert is_abstention("  I couldn't find this in your documents")
    assert not is_abstention("The answer is 4 [1].")
    sources = [Source(n, None, None, (), "", 0.0) for n in (1, 2, 3)]
    assert [s.n for s in cited_sources("See [3] and [1][1], not [7].", sources)] == [1, 3]
    assert sources[0].label == "unknown file"
