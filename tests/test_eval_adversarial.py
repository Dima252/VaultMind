import os
import re
from collections import Counter
from pathlib import Path

import pytest

from vaultmind.eval.adversarial import SUBDIR, install_adversarial
from vaultmind.eval.dataset import EvalItem, load_dataset
from vaultmind.eval.metrics import normalize_text

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "eval" / "fixtures" / "adversarial"
DATASET = ROOT / "eval" / "datasets" / "adversarial.jsonl"

INJECTION_STYLES = {"override", "fake-system", "exfiltration", "subtle"}
PAYLOAD_CODE = re.compile(r"\b[A-Z]{4,}-\d{4}\b")
URL_HOST = re.compile(r"(?i)\bhttps?://([^/\s)>\]\"']+)")
EMAIL_HOST = re.compile(r"(?i)\b[\w.+-]+@([\w-]+(?:\.[\w-]+)+)")
BARE_HOST = re.compile(r"(?i)\bwww\.[\w-]+(?:\.[\w-]+)+")


# --- install_adversarial ----------------------------------------------------------


def _fixtures(tmp_path: Path) -> Path:
    src = tmp_path / "fixtures"
    src.mkdir()
    (src / "a.md").write_text("# A\n\nalpha\n", encoding="utf-8")
    (src / "b.md").write_text("# B\n\nbeta\n", encoding="utf-8")
    (src / "notes.txt").write_text("not a fixture", encoding="utf-8")
    return src


def test_install_copies_markdown_fixtures(tmp_path: Path) -> None:
    src = _fixtures(tmp_path)
    vault = tmp_path / "vault"
    paths = install_adversarial(vault, src)
    dest = vault / SUBDIR
    assert paths == [dest / "a.md", dest / "b.md"]
    assert (dest / "a.md").read_bytes() == (src / "a.md").read_bytes()
    assert not (dest / "notes.txt").exists()


def test_install_is_idempotent_and_skips_unchanged(tmp_path: Path) -> None:
    src = _fixtures(tmp_path)
    vault = tmp_path / "vault"
    first = install_adversarial(vault, src)
    target = vault / SUBDIR / "a.md"
    # Backdate the copy so a rewrite would be visible even on coarse-grained file systems.
    old = 1_000_000_000_000_000_000
    os.utime(target, ns=(old, old))
    assert install_adversarial(vault, src) == first
    assert target.stat().st_mtime_ns == old  # unchanged bytes -> not rewritten


def test_install_rewrites_changed_fixture(tmp_path: Path) -> None:
    src = _fixtures(tmp_path)
    vault = tmp_path / "vault"
    install_adversarial(vault, src)
    (vault / SUBDIR / "a.md").write_text("tampered", encoding="utf-8")
    install_adversarial(vault, src)
    assert (vault / SUBDIR / "a.md").read_bytes() == (src / "a.md").read_bytes()


def test_install_removes_stale_files_only_in_its_folder(tmp_path: Path) -> None:
    src = _fixtures(tmp_path)
    vault = tmp_path / "vault"
    (vault / SUBDIR).mkdir(parents=True)
    (vault / SUBDIR / "old.md").write_text("stale", encoding="utf-8")
    (vault / "mine.md").write_text("user doc", encoding="utf-8")
    (vault / "cuad").mkdir()
    (vault / "cuad" / "c.txt").write_text("contract", encoding="utf-8")

    install_adversarial(vault, src)
    assert sorted(p.name for p in (vault / SUBDIR).iterdir()) == ["a.md", "b.md"]
    assert (vault / "mine.md").read_text(encoding="utf-8") == "user doc"
    assert (vault / "cuad" / "c.txt").read_text(encoding="utf-8") == "contract"


def test_install_errors_on_missing_or_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        install_adversarial(tmp_path / "vault", tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "readme.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        install_adversarial(tmp_path / "vault", empty)


def test_install_real_fixtures(tmp_path: Path) -> None:
    paths = install_adversarial(tmp_path, FIXTURES)
    assert [p.name for p in paths] == sorted(p.name for p in FIXTURES.glob("*.md"))


# --- dataset integrity against the committed fixtures ------------------------------


@pytest.fixture(scope="module")
def items() -> list[EvalItem]:
    return load_dataset(DATASET)


@pytest.fixture(scope="module")
def fixtures() -> dict[str, str]:
    """Fixture text keyed by vault-relative path (``adversarial/<name>.md``)."""
    return {f"{SUBDIR}/{p.name}": p.read_text(encoding="utf-8") for p in FIXTURES.glob("*.md")}


def _pair_prefix(source_file: str) -> str:
    """Contradiction pairs share a file-name prefix, e.g. ``adversarial/ferncliff-``."""
    return source_file.split("-", 1)[0] + "-"


def test_dataset_shape(items: list[EvalItem]) -> None:
    assert 10 <= len(items) <= 12
    for item in items:
        assert item.answerable and item.answer.strip()
        assert item.source_page is None
        assert item.evidence
        assert item.tags[0] == "adversarial"


def test_every_source_file_exists(items: list[EvalItem], fixtures: dict[str, str]) -> None:
    for item in items:
        assert item.source_file in fixtures, item.source_file


def test_evidence_appears_in_fixture(items: list[EvalItem], fixtures: dict[str, str]) -> None:
    for item in items:
        assert item.source_file is not None
        if "contradiction" in item.tags:
            prefix = _pair_prefix(item.source_file)
            texts = [t for name, t in fixtures.items() if name.startswith(prefix)]
            assert len(texts) == 2, f"expected a pair for {item.source_file}"
        else:
            texts = [fixtures[item.source_file]]
        normalized = [normalize_text(t) for t in texts]
        for span in item.evidence:
            assert any(normalize_text(span) in t for t in normalized), span


def test_contradiction_evidence_spans_both_documents(
    items: list[EvalItem], fixtures: dict[str, str]
) -> None:
    for item in items:
        if "contradiction" not in item.tags:
            continue
        assert item.source_file is not None
        prefix = _pair_prefix(item.source_file)
        pair = {n: normalize_text(t) for n, t in fixtures.items() if n.startswith(prefix)}
        hit = {n for n, t in pair.items() for s in item.evidence if normalize_text(s) in t}
        assert hit == set(pair), f"evidence must quote both documents of {prefix}"


def test_forbidden_strings_are_planted_but_never_expected(
    items: list[EvalItem], fixtures: dict[str, str]
) -> None:
    corpus = [normalize_text(t) for t in fixtures.values()]
    for item in items:
        for bad in item.forbidden:
            needle = normalize_text(bad)
            assert any(needle in t for t in corpus), f"payload not planted: {bad}"
            assert needle not in normalize_text(item.answer), bad
            for span in item.evidence:
                assert needle not in normalize_text(span), bad


def test_injection_items_carry_their_document_payload(
    items: list[EvalItem], fixtures: dict[str, str]
) -> None:
    codes_by_file: dict[str, set[str]] = {}
    for item in items:
        if "injection" not in item.tags:
            assert not item.forbidden
            continue
        assert item.source_file is not None
        codes = {f for f in item.forbidden if PAYLOAD_CODE.fullmatch(f)}
        assert len(codes) == 1, item.question
        code = codes.pop()
        assert code in fixtures[item.source_file]
        codes_by_file.setdefault(item.source_file, set()).add(code)

    # One payload code per poisoned document, unique across documents.
    assert all(len(c) == 1 for c in codes_by_file.values())
    all_codes = [next(iter(c)) for c in codes_by_file.values()]
    assert len(set(all_codes)) == len(all_codes) == 4


def test_payload_codes_appear_once_in_the_corpus(fixtures: dict[str, str]) -> None:
    found = Counter(code for text in fixtures.values() for code in PAYLOAD_CODE.findall(text))
    assert len(found) == 4
    assert all(n == 1 for n in found.values()), found


def test_tag_counts(items: list[EvalItem]) -> None:
    styles = Counter(tag for item in items if "injection" in item.tags for tag in item.tags[2:])
    assert set(styles) == INJECTION_STYLES
    assert all(n == 2 for n in styles.values())
    for item in items:
        if "injection" in item.tags:
            assert item.tags[:2] == ["adversarial", "injection"] and len(item.tags) == 3
    contradictions = [i for i in items if "contradiction" in i.tags]
    assert len(contradictions) == 3
    assert all(i.tags == ["adversarial", "contradiction"] for i in contradictions)


def _hosts(text: str) -> list[str]:
    hosts = URL_HOST.findall(text) + EMAIL_HOST.findall(text) + BARE_HOST.findall(text)
    return [h.lower().rstrip(".").split(":", 1)[0] for h in hosts]


def _allowed(host: str) -> bool:
    return (
        host.endswith(".invalid")
        or host == "invalid"
        or host == "example.com"
        or host.endswith(".example.com")
    )


def test_no_real_urls_or_email_domains(fixtures: dict[str, str]) -> None:
    texts = [*fixtures.values(), DATASET.read_text(encoding="utf-8")]
    hosts = [h for t in texts for h in _hosts(t)]
    assert hosts, "expected some placeholder addresses in the fixtures"
    assert [h for h in hosts if not _allowed(h)] == []


def test_host_check_catches_real_domains() -> None:
    assert not _allowed(_hosts("see https://evil.com/x")[0])
    assert not _allowed(_hosts("mail ops@corp.example.org")[0])
    assert _allowed(_hosts("mail ops@corp.example.com")[0])
    assert _allowed(_hosts("go to https://kestrel.invalid/upload")[0])
