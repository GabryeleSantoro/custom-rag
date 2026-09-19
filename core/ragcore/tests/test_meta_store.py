"""SQLite state: sources, documents, chats, settings. Survives a reopen."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
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


def test_a_document_can_be_fetched_by_id(tmp_path: Path) -> None:
    store = MetaStore(tmp_path / "app.db")
    source = store.add_source(SourceCreate(path="/corpus"))
    store.upsert_document(a_document("doc_1", source.id))

    fetched = store.get_document("doc_1")

    assert fetched is not None and fetched.path == "/corpus/doc_1.md"


def test_an_unknown_document_is_none_not_an_error(tmp_path: Path) -> None:
    assert MetaStore(tmp_path / "app.db").get_document("doc_nope") is None


def test_upserting_the_same_id_replaces_the_row(tmp_path: Path) -> None:
    store = MetaStore(tmp_path / "app.db")
    source = store.add_source(SourceCreate(path="/corpus"))
    store.upsert_document(a_document("doc_1", source.id))

    changed = a_document("doc_1", source.id)
    changed.status = "error"
    store.upsert_document(changed)

    assert len(store.list_documents()) == 1
    assert store.list_documents()["doc_1"].status == "error"


def test_documents_can_be_deleted_in_bulk(tmp_path: Path) -> None:
    store = MetaStore(tmp_path / "app.db")
    source = store.add_source(SourceCreate(path="/corpus"))
    for doc_id in ("doc_1", "doc_2", "doc_3"):
        store.upsert_document(a_document(doc_id, source.id))

    store.delete_documents(["doc_1", "doc_3"])

    assert list(store.list_documents()) == ["doc_2"]


def test_deleting_an_empty_list_deletes_nothing(tmp_path: Path) -> None:
    """An empty `IN ()` must never be read as "everything"."""
    store = MetaStore(tmp_path / "app.db")
    source = store.add_source(SourceCreate(path="/corpus"))
    store.upsert_document(a_document("doc_1", source.id))

    store.delete_documents([])

    assert list(store.list_documents()) == ["doc_1"]


def test_removing_a_source_with_no_documents_is_harmless(tmp_path: Path) -> None:
    store = MetaStore(tmp_path / "app.db")
    source = store.add_source(SourceCreate(path="/corpus"))

    assert store.remove_source(source.id) == []
    assert store.list_sources() == {}


def test_a_sha_can_be_updated_in_place(tmp_path: Path) -> None:
    store = MetaStore(tmp_path / "app.db")
    store.set_sha("/corpus/doc_1.md", "before")

    store.set_sha("/corpus/doc_1.md", "after")

    assert store.sha_index() == {"/corpus/doc_1.md": "after"}


def test_removing_a_source_forgets_its_shas(tmp_path: Path) -> None:
    store = MetaStore(tmp_path / "app.db")
    source = store.add_source(SourceCreate(path="/corpus"))
    store.upsert_document(a_document("doc_1", source.id))
    store.set_sha("/corpus/doc_1.md", "deadbeef")

    store.remove_source(source.id)

    assert store.sha_index() == {}


def test_wipe_clears_everything_but_the_settings(tmp_path: Path) -> None:
    db = tmp_path / "app.db"
    store = MetaStore(db)
    default = AppSettings(storage_path=str(tmp_path))
    settings = store.load_settings(default)
    settings.telemetry = True
    store.save_settings(settings)
    source = store.add_source(SourceCreate(path="/corpus"))
    store.upsert_document(a_document("doc_1", source.id))
    store.set_sha("/corpus/doc_1.md", "abc")
    session = store.create_session(None, None)
    store.append_message(
        ChatMessage(
            id="msg_1", session_id=session.id, role="user", text="hi",
            created_at=datetime.now(tz=UTC),
        )
    )

    store.wipe(keep_connections=True)

    assert store.list_sources() == {}
    assert store.list_documents() == {}
    assert store.sha_index() == {}
    assert store.list_sessions() == {}
    assert store.list_messages() == {}
    assert store.load_settings(default).telemetry is True


def test_the_first_user_message_names_an_untitled_session(tmp_path: Path) -> None:
    store = MetaStore(tmp_path / "app.db")
    session = store.create_session(None, None)

    store.append_message(
        ChatMessage(
            id="msg_1", session_id=session.id, role="user",
            text="How does reranking improve retrieval?",
            created_at=datetime.now(tz=UTC),
        )
    )

    reloaded = store.list_sessions()[session.id]
    assert reloaded.title == "How does reranking improve retrieval?"
    assert reloaded.message_count == 1


def test_a_message_for_an_unknown_session_is_stored_without_a_session_update(
    tmp_path: Path,
) -> None:
    store = MetaStore(tmp_path / "app.db")

    store.append_message(
        ChatMessage(
            id="msg_1", session_id="chat_orphan", role="user", text="hi",
            created_at=datetime.now(tz=UTC),
        )
    )

    assert store.list_sessions() == {}
    assert [m.text for m in store.list_messages()["chat_orphan"]] == ["hi"]


def test_messages_come_back_in_the_order_they_were_written(tmp_path: Path) -> None:
    store = MetaStore(tmp_path / "app.db")
    session = store.create_session("Chat", None)
    base = datetime.now(tz=UTC)
    for index, (role, text) in enumerate(
        [("user", "first"), ("assistant", "second"), ("user", "third")]
    ):
        store.append_message(
            ChatMessage(
                id=f"msg_{index}", session_id=session.id, role=role, text=text,
                created_at=base + timedelta(seconds=index),
            )
        )

    assert [m.text for m in store.list_messages()[session.id]] == ["first", "second", "third"]


def test_a_session_with_no_messages_still_appears_in_the_grouping(tmp_path: Path) -> None:
    store = MetaStore(tmp_path / "app.db")
    session = store.create_session("Empty", None)

    assert store.list_messages() == {session.id: []}


def test_a_document_scope_survives_a_reopen(tmp_path: Path) -> None:
    db = tmp_path / "app.db"
    store = MetaStore(db)
    session = store.create_session("Scoped", "doc_1")
    store.close()

    assert MetaStore(db).list_sessions()[session.id].scope_doc_id == "doc_1"


def test_settings_are_written_once_then_read_back(tmp_path: Path) -> None:
    db = tmp_path / "app.db"
    store = MetaStore(db)
    first = AppSettings(storage_path="/first")

    store.load_settings(first)

    # The second call must return what was stored, not the new default.
    assert store.load_settings(AppSettings(storage_path="/second")).storage_path == "/first"
