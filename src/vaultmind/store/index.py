"""LanceDB index of child chunks (vectors + text) and their parent sections.

Privacy: lancedb 0.40 OSS has no telemetry. Its only network code is for remote URIs
(s3://, gs://, az://, LanceDB Cloud) and opt-in OpenTelemetry; a local directory stays offline.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
from lancedb.index import FTS
from lancedb.query import LanceVectorQueryBuilder

from vaultmind.core.models import ChildChunk, ParentSection, SearchHit

_CHILDREN = "children"
_PARENTS = "parents"
_META = "meta"

_FILE_ID_RE = re.compile(r"[0-9a-fA-F:]+")
_CHUNK_ID_RE = re.compile(r"[0-9A-Za-z:_.-]+")

_PARENTS_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("file_id", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("section_path", pa.list_(pa.string())),
        pa.field("page_start", pa.int32()),
        pa.field("page_end", pa.int32()),
    ]
)

_META_SCHEMA = pa.schema(
    [
        pa.field("key", pa.string(), nullable=False),
        pa.field("value", pa.string(), nullable=False),
    ]
)

_CHILD_COLUMNS = ["id", "parent_id", "file_id", "text", "embed_text", "page", "section_path"]

# BM25 runs over `embed_text` (title > section + text), so a question that names its document
# matches that document's chunks. The simple tokenizer with the built-in English stemmer and
# stop words needs no downloaded dictionaries.
_FTS_COLUMN = "embed_text"


class EmbedderMismatchError(Exception):
    """The index was built with a different embedding model; it must be rebuilt."""


def _children_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("parent_id", pa.string(), nullable=False),
            pa.field("file_id", pa.string(), nullable=False),
            pa.field("text", pa.string(), nullable=False),
            pa.field("embed_text", pa.string(), nullable=False),
            pa.field("page", pa.int32()),
            pa.field("section_path", pa.list_(pa.string())),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
    )


def _quote_ids(ids: Sequence[str], pattern: re.Pattern[str]) -> str:
    """Render ids as a SQL list literal, refusing anything outside the safe charset."""
    for value in ids:
        if not pattern.fullmatch(value):
            raise ValueError(f"invalid id for index filter: {value!r}")
    return ", ".join(f"'{value}'" for value in ids)


class ChunkIndex:
    def __init__(self, db_dir: Path, *, embedder_name: str) -> None:
        db_dir.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(db_dir)
        tables = set(self._db.list_tables().tables)

        if _META in tables:
            self._meta = self._db.open_table(_META)
            meta = {row["key"]: row["value"] for row in self._meta.to_arrow().to_pylist()}
        else:
            self._meta = self._db.create_table(_META, schema=_META_SCHEMA)
            meta = {}

        stored = meta.get("embedder_name")
        if stored is None:
            self._set_meta("embedder_name", embedder_name)
        elif stored != embedder_name:
            raise EmbedderMismatchError(
                f"The index at {db_dir} was built with embedder {stored!r}, but the configured "
                f"embedder is {embedder_name!r}. Re-index your documents (delete the index "
                "directory and ingest again) or switch back to the original embedder."
            )

        self._dim: int | None = int(meta["dim"]) if "dim" in meta else None
        self._parents = (
            self._db.open_table(_PARENTS)
            if _PARENTS in tables
            else self._db.create_table(_PARENTS, schema=_PARENTS_SCHEMA)
        )
        self._children = self._db.open_table(_CHILDREN) if _CHILDREN in tables else None
        if self._children is not None and self._dim is None:  # meta write was interrupted
            self._dim = int(self._children.schema.field("vector").type.list_size)

    def add(
        self,
        parents: Sequence[ParentSection],
        children: Sequence[ChildChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        """Insert chunks; `vectors[i]` is the embedding of `children[i].embed_text`."""
        if len(children) != len(vectors):
            raise ValueError(f"got {len(children)} children but {len(vectors)} vectors")
        if children:
            dims = {len(v) for v in vectors}
            if len(dims) != 1:
                raise ValueError(f"vectors in one batch have different dimensions: {dims}")
            dim = dims.pop()
            self._check_dim(dim)
            if self._children is None:
                self._children = self._db.create_table(_CHILDREN, schema=_children_schema(dim))
                self._dim = dim
                self._set_meta("dim", str(dim))

        if parents:
            parent_rows: list[dict[str, Any]] = [
                {
                    "id": p.id,
                    "file_id": p.file_id,
                    "text": p.text,
                    "section_path": list(p.section_path),
                    "page_start": p.page_start,
                    "page_end": p.page_end,
                }
                for p in parents
            ]
            self._upsert(self._parents, pa.Table.from_pylist(parent_rows, schema=_PARENTS_SCHEMA))

        if children and self._children is not None:
            child_rows: list[dict[str, Any]] = [
                {
                    "id": c.id,
                    "parent_id": c.parent_id,
                    "file_id": c.file_id,
                    "text": c.text,
                    "embed_text": c.embed_text,
                    "page": c.page,
                    "section_path": list(c.section_path),
                    "vector": [float(x) for x in v],
                }
                for c, v in zip(children, vectors, strict=True)
            ]
            self._upsert(
                self._children, pa.Table.from_pylist(child_rows, schema=self._children.schema)
            )

    def delete_file(self, file_id: str) -> None:
        if not _FILE_ID_RE.fullmatch(file_id):
            raise ValueError(f"invalid file id: {file_id!r}")
        where = f"file_id = '{file_id}'"
        self._parents.delete(where)
        if self._children is not None:
            self._children.delete(where)

    def vector_search(self, vector: Sequence[float], k: int) -> list[SearchHit]:
        if self._children is None or k <= 0:
            return []
        self._check_dim(len(vector))
        query = self._children.search([float(x) for x in vector], vector_column_name="vector")
        assert isinstance(query, LanceVectorQueryBuilder)
        rows = (
            query.distance_type("cosine")
            .select([*_CHILD_COLUMNS, "_distance"])
            .limit(k)
            .to_arrow()
            .to_pylist()
        )
        hits = [SearchHit(chunk=_child_from_row(r), score=_cosine_score(r)) for r in rows]
        return sorted(hits, key=lambda h: h.score, reverse=True)

    def fts_search(self, query: str, k: int) -> list[SearchHit]:
        """BM25 keyword search over child chunks; `score` is the raw BM25 score.

        The query is plain text (no query syntax). Rows added since the last `refresh_fts`
        are still searched, just without the index.
        """
        if self._children is None or k <= 0 or not query.strip():
            return []
        if not self._has_fts():
            self.refresh_fts()
        rows = (
            self._children.search(query, query_type="fts", fts_columns=_FTS_COLUMN)
            .select([*_CHILD_COLUMNS, "_score"])
            .limit(k)
            .to_arrow()
            .to_pylist()
        )
        hits = [SearchHit(chunk=_child_from_row(r), score=float(r["_score"])) for r in rows]
        return sorted(hits, key=lambda h: h.score, reverse=True)

    def refresh_fts(self) -> None:
        """(Re)build the full-text index over all current child chunks."""
        if self._children is not None and self._children.count_rows() > 0:
            self._children.create_index(_FTS_COLUMN, config=FTS(), replace=True)

    def get_parents(self, parent_ids: Sequence[str]) -> dict[str, ParentSection]:
        if not parent_ids:
            return {}
        where = f"id IN ({_quote_ids(list(dict.fromkeys(parent_ids)), _CHUNK_ID_RE)})"
        rows = self._parents.search().where(where).limit(None).to_arrow().to_pylist()
        return {r["id"]: _parent_from_row(r) for r in rows}

    def parents_for_file(self, file_id: str) -> list[ParentSection]:
        """Every parent section of a file, in document order (`[]` for an unknown file)."""
        if not _FILE_ID_RE.fullmatch(file_id):
            raise ValueError(f"invalid file id: {file_id!r}")
        rows = (
            self._parents.search()
            .where(f"file_id = '{file_id}'")
            .limit(None)
            .to_arrow()
            .to_pylist()
        )
        return sorted((_parent_from_row(r) for r in rows), key=_parent_order)

    def close(self) -> None:
        """Release the index. A no-op today (LanceDB holds no open handles); mirrors Registry."""

    def count(self) -> int:
        """Number of child chunks in the index."""
        return 0 if self._children is None else int(self._children.count_rows())

    def _has_fts(self) -> bool:
        assert self._children is not None
        return any(ix.index_type == "FTS" for ix in self._children.list_indices())

    def _check_dim(self, dim: int) -> None:
        if self._dim is not None and dim != self._dim:
            raise ValueError(
                f"vector dimension {dim} does not match the index dimension {self._dim}"
            )

    def _set_meta(self, key: str, value: str) -> None:
        row = pa.Table.from_pylist([{"key": key, "value": value}], _META_SCHEMA)
        self._upsert(self._meta, row, key="key")

    @staticmethod
    def _upsert(table: Any, data: pa.Table, key: str = "id") -> None:
        """Insert rows, replacing any existing row with the same key."""
        (
            table.merge_insert(key)
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(data)
        )


def _cosine_score(row: dict[str, Any]) -> float:
    """LanceDB returns cosine distance (1 - similarity); clamp away float noise."""
    similarity = 1.0 - float(row["_distance"])
    if math.isnan(similarity):  # a zero vector has no direction
        return 0.0
    return max(-1.0, min(1.0, similarity))


def _child_from_row(row: dict[str, Any]) -> ChildChunk:
    return ChildChunk(
        id=row["id"],
        parent_id=row["parent_id"],
        file_id=row["file_id"],
        text=row["text"],
        embed_text=row["embed_text"],
        page=row["page"],
        section_path=tuple(row["section_path"] or ()),
    )


def _parent_order(parent: ParentSection) -> int:
    """Position in the document, from the id's `:p{index}` suffix."""
    return int(parent.id.rsplit(":p", 1)[1])


def _parent_from_row(row: dict[str, Any]) -> ParentSection:
    return ParentSection(
        id=row["id"],
        file_id=row["file_id"],
        text=row["text"],
        section_path=tuple(row["section_path"] or ()),
        page_start=row["page_start"],
        page_end=row["page_end"],
    )
