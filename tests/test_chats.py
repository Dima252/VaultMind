import sqlite3
import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from vaultmind.agent.baseline import Source
from vaultmind.agent.types import AgentResult, Contradiction, UnsupportedClaim
from vaultmind.core.models import FileRecord
from vaultmind.store.chats import DEFAULT_TITLE, ChatMessage, ChatStore
from vaultmind.store.registry import SCHEMA_VERSION, Registry
from vaultmind.store.traces import TraceStore

HEBREW_Q = "מה תקופת ההודעה המוקדמת?"
HEBREW_A = "שלושים יום [1]."

SOURCES = [
    Source(
        n=1,
        file_path=Path("vault") / "חוזה שכירות.pdf",
        page=3,
        section_path=("פרק א", "הודעה"),
        text="תקופת ההודעה היא שלושים יום.",
        score=0.8734512345678901,
        chunk_id="c1",
        warnings=("instruction-like text",),
    ),
    Source(
        n=2,
        file_path=None,
        page=None,
        section_path=(),
        text="Plain text without a page.",
        score=-1.5,
    ),
]


def _result(
    question: str = "What is the notice period?",
    answer: str = "Thirty days [1].",
    *,
    sources: list[Source] | None = None,
    abstained: bool = False,
    contradictions: list[Contradiction] | None = None,
    unsupported: list[UnsupportedClaim] | None = None,
) -> AgentResult:
    return AgentResult(
        question=question,
        answer=answer,
        sources=list(sources or []),
        abstained=abstained,
        queries=[question],
        rewrites=0,
        contradictions=list(contradictions or []),
        unsupported=list(unsupported or []),
    )


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "vaultmind.sqlite"


@pytest.fixture
def store(db_path: Path) -> Iterator[ChatStore]:
    s = ChatStore(db_path)
    yield s
    s.close()


def test_create_and_get(store: ChatStore) -> None:
    before = datetime.now(UTC)
    chat = store.create()
    assert len(chat.id) == 32 and int(chat.id, 16) >= 0
    assert chat.title == DEFAULT_TITLE
    assert chat.created_at.tzinfo is not None
    assert before - timedelta(seconds=1) <= chat.created_at <= datetime.now(UTC)
    assert store.get(chat.id) == chat
    assert store.create("Lease").title == "Lease"


def test_unknown_chat(store: ChatStore) -> None:
    assert store.get("nope") is None
    assert store.messages("nope") == []
    assert store.history("nope") == []
    with pytest.raises(KeyError):
        store.rename("nope", "x")
    with pytest.raises(KeyError):
        store.add_exchange("nope", _result(), None)
    store.delete("nope")  # no-op


def test_recent_most_recently_updated_first(store: ChatStore) -> None:
    a, b, c = store.create("a"), store.create("b"), store.create("c")
    assert [x.id for x in store.recent()] == [c.id, b.id, a.id]
    store.add_exchange(a.id, _result(), None)
    assert [x.id for x in store.recent()] == [a.id, c.id, b.id]
    assert [x.id for x in store.recent(limit=2)] == [a.id, c.id]
    updated = store.get(a.id)
    assert updated is not None and updated.updated_at > a.updated_at
    assert updated.created_at == a.created_at


def test_rename(store: ChatStore) -> None:
    chat = store.create()
    store.rename(chat.id, "Renamed")
    got = store.get(chat.id)
    assert got is not None and got.title == "Renamed"
    assert got.updated_at == chat.updated_at  # renaming isn't a new message


def test_add_exchange_seq_and_return(store: ChatStore) -> None:
    chat = store.create()
    user, assistant = store.add_exchange(chat.id, _result(), None)
    assert (user.seq, user.role, user.content) == (0, "user", "What is the notice period?")
    assert (assistant.seq, assistant.role, assistant.content) == (
        1,
        "assistant",
        "Thirty days [1].",
    )
    assert user.run_id is None and user.sources == () and not user.abstained
    user2, assistant2 = store.add_exchange(chat.id, _result("And then?", "Nothing."), None)
    assert (user2.seq, assistant2.seq) == (2, 3)
    assert store.messages(chat.id) == [user, assistant, user2, assistant2]


def test_auto_title_only_first_exchange_and_default_title(store: ChatStore) -> None:
    chat = store.create()
    store.add_exchange(chat.id, _result("  What is   the notice period?  "), None)
    store.add_exchange(chat.id, _result("A second question"), None)
    got = store.get(chat.id)
    assert got is not None and got.title == "What is the notice period?"

    named = store.create("My title")
    store.add_exchange(named.id, _result(), None)
    got = store.get(named.id)
    assert got is not None and got.title == "My title"

    # Renamed back to the default after an exchange: no longer the first one, so no auto-title.
    later = store.create()
    store.add_exchange(later.id, _result(), None)
    store.rename(later.id, DEFAULT_TITLE)
    store.add_exchange(later.id, _result("Other"), None)
    got = store.get(later.id)
    assert got is not None and got.title == DEFAULT_TITLE


def test_auto_title_long_multiline_and_blank(store: ChatStore) -> None:
    long_q = "Please summarize every obligation of the tenant under the lease agreement in detail"
    chat = store.create()
    store.add_exchange(chat.id, _result(long_q), None)
    got = store.get(chat.id)
    assert got is not None
    assert got.title == "Please summarize every obligation of the tenant under the..."
    assert len(got.title) <= 63

    multi = store.create()
    store.add_exchange(multi.id, _result("\n\n  First line\twith tab \nsecond line"), None)
    got = store.get(multi.id)
    assert got is not None and got.title == "First line with tab"

    hebrew = store.create()
    store.add_exchange(hebrew.id, _result(HEBREW_Q, HEBREW_A), None)
    got = store.get(hebrew.id)
    assert got is not None and got.title == HEBREW_Q

    blank = store.create()
    store.add_exchange(blank.id, _result("   \n "), None)
    got = store.get(blank.id)
    assert got is not None and got.title == DEFAULT_TITLE

    unbroken = store.create()
    store.add_exchange(unbroken.id, _result("x" * 100), None)
    got = store.get(unbroken.id)
    assert got is not None and got.title == "x" * 60 + "..."


def test_assistant_round_trip(db_path: Path) -> None:
    contradictions = [Contradiction("30 days", 1, "60 days", 2)]
    unsupported = [
        UnsupportedClaim("The fee is 5%.", (1, 2), "no source mentions a fee"),
        UnsupportedClaim("Uncited claim.", (), ""),
    ]
    result = _result(
        HEBREW_Q,
        HEBREW_A,
        sources=SOURCES,
        contradictions=contradictions,
        unsupported=unsupported,
    )
    store = ChatStore(db_path)
    try:
        chat = store.create()
        user, assistant = store.add_exchange(chat.id, result, None)
    finally:
        store.close()
    reopened = ChatStore(db_path)  # a fresh connection reads only what was stored
    try:
        got_user, got_assistant = reopened.messages(chat.id)
    finally:
        reopened.close()
    assert got_user == user
    assert got_assistant == assistant
    assert got_assistant.sources == tuple(SOURCES)
    assert got_assistant.contradictions == tuple(contradictions)
    assert got_assistant.unsupported == tuple(unsupported)
    assert got_assistant.created_at.tzinfo is not None
    assert not got_assistant.abstained


def test_abstained_round_trip(store: ChatStore) -> None:
    chat = store.create()
    result = _result(
        answer="I couldn't find this in your documents.", sources=SOURCES[:1], abstained=True
    )
    _, assistant = store.add_exchange(chat.id, result, None)
    (got,) = [m for m in store.messages(chat.id) if m.role == "assistant"]
    assert got.abstained and got == assistant


def test_user_message_meta_is_empty(store: ChatStore, db_path: Path) -> None:
    chat = store.create()
    store.add_exchange(chat.id, _result(sources=SOURCES), None)
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT role, meta FROM messages ORDER BY seq").fetchall()
    finally:
        conn.close()
    assert rows[0] == ("user", "{}")
    assert rows[1][0] == "assistant" and '"sources"' in rows[1][1]
    assert "חוזה" in rows[1][1]  # stored as UTF-8 text, not \u escapes


def test_history(store: ChatStore) -> None:
    chat = store.create()
    store.add_exchange(chat.id, _result("q1", "a1 [1]."), None)
    store.add_exchange(chat.id, _result(HEBREW_Q, HEBREW_A), None)
    assert store.history(chat.id) == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1 [1]."},
        {"role": "user", "content": HEBREW_Q},
        {"role": "assistant", "content": HEBREW_A},
    ]


def test_delete_cascades_messages_keeps_runs(db_path: Path) -> None:
    traces = TraceStore(db_path)
    store = ChatStore(db_path)
    try:
        result = _result()
        run_id = traces.save(result)
        chat = store.create()
        other = store.create()
        _, assistant = store.add_exchange(chat.id, result, run_id)
        store.add_exchange(other.id, result, None)
        assert assistant.run_id == run_id
        assert store.messages(chat.id)[1].run_id == run_id

        store.delete(chat.id)
        assert store.get(chat.id) is None
        assert store.messages(chat.id) == []
        assert len(store.messages(other.id)) == 2
        assert traces.get(run_id) is not None
    finally:
        store.close()
        traces.close()

    conn = sqlite3.connect(db_path)
    try:
        orphans = conn.execute("SELECT COUNT(*) FROM messages WHERE chat_id = ?", (chat.id,))
        assert orphans.fetchone()[0] == 0
    finally:
        conn.close()


def test_failed_exchange_leaves_nothing(store: ChatStore) -> None:
    chat = store.create()
    with pytest.raises(sqlite3.IntegrityError):  # run id not in `runs`
        store.add_exchange(chat.id, _result(), "missing-run")
    assert store.messages(chat.id) == []
    got = store.get(chat.id)
    assert got == chat  # title and updated_at untouched


def test_coexists_with_registry_and_traces(db_path: Path) -> None:
    registry = Registry(db_path)
    traces = TraceStore(db_path)
    store = ChatStore(db_path)
    try:
        registry.upsert(FileRecord(id="f1", path=Path("a.pdf"), sha256="x", status="ready"))
        run_id = traces.save(_result())
        chat = store.create()
        store.add_exchange(chat.id, _result(), run_id)
        assert registry.get("f1") is not None
        assert traces.get(run_id) is not None
        assert len(store.messages(chat.id)) == 2
    finally:
        for s in (store, traces, registry):
            s.close()
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        conn.close()


def test_concurrent_exchanges_get_distinct_seqs(db_path: Path) -> None:
    stores = [ChatStore(db_path), ChatStore(db_path)]  # two connections, like two threads' stores
    try:
        chat = stores[0].create()

        def work(s: ChatStore) -> None:
            for i in range(10):
                s.add_exchange(chat.id, _result(f"q{i}", f"a{i}"), None)

        threads = [threading.Thread(target=work, args=(s,)) for s in stores for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        messages = stores[0].messages(chat.id)
        assert [m.seq for m in messages] == list(range(80))
        assert [m.role for m in messages] == ["user", "assistant"] * 40
    finally:
        for s in stores:
            s.close()


def test_close_is_idempotent(db_path: Path) -> None:
    store = ChatStore(db_path)
    store.close()
    store.close()
    with pytest.raises(RuntimeError):
        store.create()


def test_message_defaults() -> None:
    m = ChatMessage("c", 0, "user", "hi", datetime.now(UTC))
    assert m.run_id is None and m.sources == () and m.contradictions == () and m.unsupported == ()
