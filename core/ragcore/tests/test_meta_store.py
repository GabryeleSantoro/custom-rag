"""SQLite state: sources, documents, chats, settings. Survives a reopen."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ragcore.api.schemas import AppSettings, ChatMessage, Document, SourceCreate
from ragcore.store.meta import MetaStore


def a_document(doc_id: str, source_id: str, sha: str = "abc") -> Document:
    return Document(
        id=doc_id,
        source_id=source_id,
        path=f"/corpus/{doc_id}.md",
        title=doc_id,
        ext=".md",
        mime="text/markdown",
        size_bytes=1024,
        n_pages=2,
        n_chunks=4,
        status="indexed",
        mtime=datetime.now(tz=UTC),
    )


def test_source_round_trips(tmp_path: Path) -> None:
    store = MetaStore(tmp_path / "app.db")
    source = store.add_source(SourceCreate(path="/corpus", include_globs=["**/*.md"]))

    assert store.list_sources()[source.id].path == "/corpus"
    assert store.list_sources()[source.id].include_globs == ["**/*.md"]


def test_state_survives_a_reopen(tmp_path: Path) -> None:
    db = tmp_path / "app.db"
    store = MetaStore(db)
    source = store.add_source(SourceCreate(path="/corpus"))
    store.upsert_document(a_document("doc_1", source.id))
    store.close()

    reopened = MetaStore(db)

    assert "doc_1" in reopened.list_documents()
    assert reopened.list_documents()["doc_1"].title == "doc_1"


def test_removing_a_source_returns_its_document_ids(tmp_path: Path) -> None:
    store = MetaStore(tmp_path / "app.db")
    source = store.add_source(SourceCreate(path="/corpus"))
    store.upsert_document(a_document("doc_1", source.id))
    store.upsert_document(a_document("doc_2", source.id))

    removed = store.remove_source(source.id)

    assert sorted(removed) == ["doc_1", "doc_2"]
    assert store.list_documents() == {}
    assert store.list_sources() == {}


def test_sha_index_maps_path_to_digest(tmp_path: Path) -> None:
    store = MetaStore(tmp_path / "app.db")
    source = store.add_source(SourceCreate(path="/corpus"))
    store.upsert_document(a_document("doc_1", source.id))
    store.set_sha("/corpus/doc_1.md", "deadbeef")

    assert store.sha_index()["/corpus/doc_1.md"] == "deadbeef"


def test_messages_are_grouped_by_session(tmp_path: Path) -> None:
    store = MetaStore(tmp_path / "app.db")
    session = store.create_session("first question", None)
    store.append_message(
        ChatMessage(
            id="msg_1",
            session_id=session.id,
            role="user",
            text="what is reranking",
            created_at=datetime.now(tz=UTC),
        )
    )

    assert [m.text for m in store.list_messages()[session.id]] == ["what is reranking"]


def test_settings_round_trip(tmp_path: Path) -> None:
    db = tmp_path / "app.db"
    store = MetaStore(db)
    default = AppSettings(storage_path=str(tmp_path))
    settings = store.load_settings(default)
    settings.retrieval.top_k = 9
    store.save_settings(settings)
    store.close()

    assert MetaStore(db).load_settings(default).retrieval.top_k == 9
