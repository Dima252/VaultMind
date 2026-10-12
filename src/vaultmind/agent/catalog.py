"""The document catalog: which files the vault has, what each is, and which one a request means.

Backed by the registry (ready files) and the document cards. The planner reads `describe()`;
the tools and recipes turn a mention ("the Workato worksheet", "הסילבוס") into a file with
`resolve()`.

Resolution goes from cheap and certain to expensive and fuzzy, and stops at the first stage
that decides: exact file id, exact name or stem, word overlap with the name, card title and
type, then the reranker (multilingual, so a Hebrew mention can find an English name). A stage
that sees several close candidates raises `AmbiguousDocument` rather than guessing.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path, PurePosixPath

from vaultmind.agent.types import AmbiguousDocument, DocumentNotFound
from vaultmind.core.models import DocCard, FileRecord
from vaultmind.core.providers import Reranker
from vaultmind.store.cards import CardStore
from vaultmind.store.registry import Registry

# Word overlap: the share of the mention's content words found among the document's words
# (name, card title, type). "the AI syllabus" covers "AI in the workplace syllabus" fully
# and "Data structures syllabus" half.
FUZZY_MIN = 0.66  # at least two of three content words
FUZZY_PARTIAL = 0.5  # half the words: offered back as "did you mean", never picked silently
FUZZY_MARGIN = 0.2  # the winner must beat the runner-up by this much
# Reranker (bge-reranker-v2-m3, sigmoid in [0, 1]) of the mention against each document's
# name and description. Measured on 8 synthetic English and Hebrew mentions of 4 documents
# (2026-10-12): the right document scored 0.49-0.99, every wrong one, and every mention of a
# document that isn't there, 0.05 or less.
RERANK_MIN = 0.15
RERANK_MARGIN = 0.15
MAX_CANDIDATES = 5

_WORD = re.compile(r"\w+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s|\n")
_STOPWORDS = frozenset(
    (
        *("a", "an", "the", "of", "my", "our", "your", "this", "that", "these", "those"),
        *("in", "on", "for", "about", "to", "and", "with", "from"),
        *("doc", "docs", "document", "documents", "file", "files", "pdf", "docx", "md", "txt"),
        *("של", "את", "על", "עם"),  # of, (object), on, with
        *("קובץ", "הקובץ", "מסמך", "המסמך"),  # file, document
    )
)
_HEBREW_PREFIXES = "הובלמשכ"


@dataclass(frozen=True)
class CatalogEntry:
    file: FileRecord
    name: str  # path relative to the vault, forward slashes (absolute if outside it)
    card: DocCard | None  # the newest card for the file's content, if one was written


class Catalog:
    def __init__(
        self,
        registry: Registry,
        cards: CardStore,
        vault_dir: Path,
        *,
        reranker: Reranker | None = None,
    ) -> None:
        self._registry = registry
        self._cards = cards
        self._vault = vault_dir.resolve()
        self._reranker = reranker

    def entries(self) -> list[CatalogEntry]:
        """Ready files, sorted by name. Read fresh on every call (files come and go)."""
        found = [
            CatalogEntry(record, self._name(record.path), self._cards.latest(record.sha256))
            for record in self._registry.all()
            if record.status == "ready"
        ]
        return sorted(found, key=lambda e: (e.name.casefold(), e.name))

    def resolve(self, mention: str) -> CatalogEntry:
        """The one document `mention` means: a file id, a file name or path, a title, or a
        description ("the AI syllabus"), in any language. Raises `DocumentNotFound` when
        nothing matches and `AmbiguousDocument` when several match about equally well."""
        text = mention.strip().strip("\"'")
        entries = self.entries()
        if not text or not entries:
            raise DocumentNotFound(mention)

        for entry in entries:
            if entry.file.id == text:
                return entry
        exact = [e for e in entries if _exact(text, e)]
        if exact:
            return _one(mention, exact)

        words = _content_words(text)
        if not words:  # "the document": fine when there's only one
            if len(entries) == 1:
                return entries[0]
            raise AmbiguousDocument(mention, tuple(e.name for e in entries[:MAX_CANDIDATES]))

        scored = _ranked([(e, _coverage(words, _entry_words(e))) for e in entries])
        best = scored[0][1]
        if best >= FUZZY_MIN:
            return _decide(mention, scored, FUZZY_MARGIN, floor=FUZZY_MIN)

        if self._reranker is not None:
            scores = _rerank_scores(self._reranker, text, entries)
            reranked = _ranked(list(zip(entries, scores, strict=True)))
            if reranked[0][1] >= RERANK_MIN:
                return _decide(mention, reranked, RERANK_MARGIN, floor=RERANK_MIN)

        partial = [e for e, score in scored if score >= FUZZY_PARTIAL]
        if partial:  # half the words match: let the user confirm instead of guessing
            raise AmbiguousDocument(mention, tuple(e.name for e in partial[:MAX_CANDIDATES]))
        raise DocumentNotFound(mention)

    def describe(self, limit: int) -> str:
        """One line per document for the planner prompt (name, type, language, a short
        summary), at most `limit` lines; the rest are counted, not listed."""
        entries = self.entries()
        if not entries:
            return "No documents are indexed yet."
        lines = [summary_line(e) for e in entries[: max(limit, 0)]]
        if len(entries) > len(lines):
            lines.append(f"... and {len(entries) - len(lines)} more")
        return "\n".join(lines)

    def _name(self, path: Path) -> str:
        for candidate in (path, path.resolve()):  # registry paths are already resolved
            if candidate.is_relative_to(self._vault):
                return candidate.relative_to(self._vault).as_posix()
        return path.as_posix()


def summary_line(entry: CatalogEntry) -> str:
    """`name | type | language | first summary sentence`, or the name alone without a card."""
    card = entry.card
    if card is None:
        return entry.name
    return " | ".join([entry.name, card.doc_type, card.language, first_sentence(card.summary)])


def first_sentence(text: str, limit: int = 200) -> str:
    sentence = _SENTENCE_END.split(text.strip(), maxsplit=1)[0].strip()
    return sentence if len(sentence) <= limit else sentence[: limit - 3].rstrip() + "..."


def _one(mention: str, matches: Sequence[CatalogEntry]) -> CatalogEntry:
    if len(matches) == 1:
        return matches[0]
    raise AmbiguousDocument(mention, tuple(e.name for e in matches[:MAX_CANDIDATES]))


def _decide(
    mention: str, ranked: Sequence[tuple[CatalogEntry, float]], margin: float, *, floor: float
) -> CatalogEntry:
    """The best entry when it clearly wins; otherwise the close ones are ambiguous."""
    best = ranked[0][1]
    close = [e for e, score in ranked if score >= floor and score > best - margin]
    return _one(mention, close)


def _ranked(scored: list[tuple[CatalogEntry, float]]) -> list[tuple[CatalogEntry, float]]:
    return sorted(scored, key=lambda pair: pair[1], reverse=True)  # stable: ties keep name order


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold().replace("\\", "/").strip()


def _exact(text: str, entry: CatalogEntry) -> bool:
    """The mention is the file's name or stem, with or without its folders."""
    wanted = _norm(text)
    path = PurePosixPath(_norm(entry.name))
    forms = {str(path), path.name, path.stem, str(path.with_suffix(""))}
    if entry.card is not None:
        forms.add(_norm(entry.card.title))
    return wanted in forms


def _words(text: str) -> list[str]:
    return _WORD.findall(_norm(text).replace("_", " "))


def _content_words(text: str) -> list[str]:
    return [w for w in _words(text) if w not in _STOPWORDS]


def _entry_words(entry: CatalogEntry) -> set[str]:
    words = set(_words(str(PurePosixPath(entry.name).with_suffix(""))))
    if entry.card is not None:
        words.update(_words(entry.card.title))
        words.update(_words(entry.card.doc_type))
    return words - _STOPWORDS


def _coverage(mention: Sequence[str], words: set[str]) -> float:
    """Mean over the mention's words of their best match among the document's words."""
    if not mention or not words:
        return 0.0
    return sum(max(_word_similarity(m, w) for w in words) for m in mention) / len(mention)


def _word_similarity(a: str, b: str) -> float:
    """1 for the same word; high for typos, Hebrew prefixes and close forms; else 0."""
    if a == b:
        return 1.0
    if min(len(a), len(b)) < 4:  # "ai" must match exactly
        return 0.0
    if (a[0] in _HEBREW_PREFIXES and a[1:] == b) or (b[0] in _HEBREW_PREFIXES and b[1:] == a):
        return 0.95  # "הסילבוס" is "the syllabus"
    short, long = sorted((a, b), key=len)
    if long.startswith(short) and (len(short) >= 5 or len(short) / len(long) >= 0.6):
        return 0.9  # "contract" / "contracts", "ferroglobe" / "ferroglobeplc"
    ratio = SequenceMatcher(None, a, b).ratio()
    return ratio if ratio >= 0.8 else 0.0


def _rerank_scores(
    reranker: Reranker, mention: str, entries: Sequence[CatalogEntry]
) -> list[float]:
    """Per entry, the better of two scores: against the bare name, and against "name | title |
    type | first summary line". Each wins on different mentions (a summary can drown a name
    that matched, and a name alone misses what only the card says), so one batch scores both."""
    names = [_bare_name(e) for e in entries]
    described = [e for e in entries if e.card is not None]
    scores = reranker.score(mention, names + [_description(e) for e in described])
    best = dict(zip((e.file.id for e in entries), scores[: len(entries)], strict=True))
    for entry, score in zip(described, scores[len(entries) :], strict=True):
        best[entry.file.id] = max(best[entry.file.id], score)
    return [best[e.file.id] for e in entries]


def _bare_name(entry: CatalogEntry) -> str:
    return str(PurePosixPath(entry.name).with_suffix("")).replace("_", " ")


def _description(entry: CatalogEntry) -> str:
    """Name, card title, type and first summary line."""
    parts = [_bare_name(entry)]
    if entry.card is not None:
        card = entry.card
        parts += [card.title, card.doc_type, first_sentence(card.summary)]
    return " | ".join(p for p in parts if p)
