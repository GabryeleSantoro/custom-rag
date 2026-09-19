"""The in-memory store behind the stub backend, and the corpus loader under it."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from ragcore.api.schemas import ChatMessage, SourceCreate
from ragcore.config import Config
from ragcore.stub.corpus import find_fixture_dir, load_corpus, load_document
from ragcore.stub.store import Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(Config(data_dir=tmp_path))


def a_message(session_id: str, role: str, text: str) -> ChatMessage:
    return ChatMessage(
        id=f"msg_{role}", session_id=session_id, role=role, text=text,
        created_at=datetime.now(tz=UTC),
    )


# ----------------------------------------------------------------- the fixtures


def test_the_fixture_corpus_is_found_from_inside_the_package() -> None:
    fixtures = find_fixture_dir()

    assert fixtures is not None
    assert fixtures.is_dir()
    assert list(fixtures.glob("*.md"))


def test_a_directory_with_no_fixtures_above_it_finds_nothing(tmp_path: Path) -> None:
    assert find_fixture_dir(tmp_path / "somewhere" / "deep") is None


def test_a_new_store_seeds_itself_from_the_fixtures(store: Store) -> None:
    assert len(store.sources) == 1
    assert store.documents
    assert all(doc.status == "indexed" for doc in store.documents.values())
    assert store.retriever.chunks


def test_a_new_store_ships_an_active_embedder_and_reranker(store: Store) -> None:
    active = {m.role for m in store.models.values() if m.active}

    assert active == {"embedding", "reranking"}
    assert all(m.shipped for m in store.models.values())


# -------------------------------------------------------------- corpus loading


def test_loading_a_document_splits_pages_on_level_two_headings(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text("# Guide\n\n## First\n\nAlpha body.\n\n## Second\n\nBeta body.\n")

    loaded = load_document(path)

    # The level-one title line is the document's own top-level section, page 1.
    assert [page.section_path for page in loaded.pages] == [
        "Guide",
        "Guide > First",
        "Guide > Second",
    ]
    assert [page.page for page in loaded.pages] == [1, 2, 3]
    assert (loaded.doc_id, loaded.title) == ("guide", "Guide")
    assert loaded.ext == ".md"
    assert loaded.mime == "text/markdown"


def test_chunk_offsets_point_back_into_their_page(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text("# Guide\n\n## First\n\nAlpha body.\n\nSecond paragraph.\n")

    loaded = load_document(path)

    pages = {page.page: page.text for page in loaded.pages}
    for chunk in loaded.chunks:
        assert pages[chunk.page][chunk.char_start : chunk.char_end].strip() == chunk.text


def test_a_heading_line_is_not_repeated_in_its_pages_body(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text("# Guide\n\n## First\n\nAlpha body.\n")

    loaded = load_document(path)

    assert loaded.pages[1].text == "Alpha body."


def test_a_document_that_starts_at_a_heading_has_no_preamble_page(tmp_path: Path) -> None:
    path = tmp_path / "noh1.md"
    path.write_text("## First\n\nAlpha body.\n")

    loaded = load_document(path)

    assert [page.section_path for page in loaded.pages] == ["noh1 > First"]
    # No level-one heading, so the filename stem is the title.
    assert loaded.title == "noh1"


def test_a_file_with_no_usable_text_loads_with_no_pages(tmp_path: Path) -> None:
    path = tmp_path / "hollow.md"
    path.write_text("## First\n\n## Second\n")

    loaded = load_document(path)

    assert loaded.pages == []
    assert loaded.chunks == []


def test_keywords_are_the_commonest_long_words(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text(
        "# Guide\n\n## First\n\nReranking reranking reranking improves retrieval of the text.\n"
    )

    loaded = load_document(path)

    assert loaded.keywords[0] == "reranking"
    assert "the" not in loaded.keywords, "short words carry no signal"
    assert len(loaded.keywords) <= 6


def test_load_corpus_honours_include_globs(tmp_path: Path) -> None:
    (tmp_path / "keep.md").write_text("# Keep\n\nA.\n")
    (tmp_path / "skip.txt").write_text("Skip me.")

    loaded = load_corpus(tmp_path, include_globs=["**/*.md"])

    assert [doc.doc_id for doc in loaded] == ["keep"]


def test_load_corpus_honours_exclude_globs(tmp_path: Path) -> None:
    (tmp_path / "keep.md").write_text("# Keep\n\nA.\n")
    (tmp_path / "draft.md").write_text("# Draft\n\nB.\n")

    loaded = load_corpus(tmp_path, include_globs=["**/*.md"], exclude_globs=["**/draft.md"])

    assert [doc.doc_id for doc in loaded] == ["keep"]


def test_load_corpus_without_globs_scans_every_supported_file(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\nAlpha.\n")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.txt").write_text("Beta.")
    (tmp_path / "c.rtf").write_text("ignored")

    loaded = load_corpus(tmp_path)

    assert sorted(doc.doc_id for doc in loaded) == ["a", "b"]


def test_load_corpus_skips_a_file_over_the_size_cap(tmp_path: Path) -> None:
    (tmp_path / "huge.md").write_text("# Huge\n\n" + "x" * 2_000_000)

    assert load_corpus(tmp_path, include_globs=["**/*.md"], max_file_mb=1) == []


# --------------------------------------------------------------------- sources


def test_ingesting_an_unknown_source_returns_nothing(store: Store) -> None:
    assert store.ingest_source("src_nope") == []


def test_ingesting_a_path_that_is_not_a_directory_returns_nothing(
    store: Store, tmp_path: Path
) -> None:
    file_path = tmp_path / "a-file.md"
    file_path.write_text("# A\n\nBody.\n")
    source = store.add_source(SourceCreate(path=str(file_path)))

    assert store.ingest_source(source.id) == []


def test_ingesting_twice_does_not_duplicate_documents(store: Store, tmp_path: Path) -> None:
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "one.md").write_text("# One\n\nBody.\n")
    source = store.add_source(SourceCreate(path=str(folder), include_globs=["**/*.md"]))

    store.ingest_source(source.id)
    store.ingest_source(source.id)

    assert len([d for d in store.documents.values() if d.source_id == source.id]) == 1
    assert store.sources[source.id].document_count == 1


def test_removing_a_source_reports_how_many_documents_went_with_it(
    store: Store, tmp_path: Path
) -> None:
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "one.md").write_text("# One\n\nBody.\n")
    (folder / "two.md").write_text("# Two\n\nBody.\n")
    source = store.add_source(SourceCreate(path=str(folder), include_globs=["**/*.md"]))
    store.ingest_source(source.id)

    assert store.remove_source(source.id) == 2
    assert source.id not in store.sources


def test_removing_an_unknown_source_removes_nothing(store: Store) -> None:
    assert store.remove_source("src_nope") == 0


# ------------------------------------------------------------------- documents


def test_content_for_an_unknown_document_is_none(store: Store) -> None:
    assert store.content("doc_nope") is None


def test_doc_meta_carries_what_the_filters_need(store: Store) -> None:
    doc_id, meta = next(iter(store.doc_meta().items()))

    assert set(meta) == {"source_id", "ext", "lang", "mtime"}
    assert meta["source_id"] == store.documents[doc_id].source_id


def test_index_stats_count_what_was_loaded(store: Store) -> None:
    stats = store.index_stats()

    assert stats.documents == len(store.documents)
    assert stats.chunks == sum(len(doc.chunks) for doc in store.loaded.values())
    assert stats.parents == sum(len(doc.pages) for doc in store.loaded.values())
    assert stats.embed_dim == 1024
    assert stats.last_indexed_at is not None


def test_index_stats_on_an_empty_store_report_nothing_indexed(store: Store) -> None:
    store.documents.clear()
    store.loaded.clear()

    stats = store.index_stats()

    assert (stats.documents, stats.chunks, stats.size_bytes) == (0, 0, 0)
    assert stats.last_indexed_at is None


# ----------------------------------------------------------------------- chats


def test_the_first_user_message_names_an_untitled_session(store: Store) -> None:
    session = store.create_session(None)

    store.append_message(a_message(session.id, "user", "How does reranking improve retrieval?"))

    assert session.title == "How does reranking improve retrieval?"
    assert session.message_count == 1


def test_a_named_session_keeps_its_name(store: Store) -> None:
    session = store.create_session("Retrieval notes")

    store.append_message(a_message(session.id, "user", "Anything"))

    assert session.title == "Retrieval notes"


def test_a_very_long_first_question_is_truncated_for_the_title(store: Store) -> None:
    session = store.create_session(None)

    store.append_message(a_message(session.id, "user", "word " * 40))

    assert len(session.title) <= 48


def test_an_assistant_message_never_titles_the_session(store: Store) -> None:
    session = store.create_session(None)

    store.append_message(a_message(session.id, "assistant", "Here is an answer."))

    assert session.title == "New chat"


def test_a_message_for_an_unknown_session_is_kept_without_crashing(store: Store) -> None:
    store.append_message(a_message("chat_orphan", "user", "Hello"))

    assert [m.text for m in store.messages["chat_orphan"]] == ["Hello"]


def test_a_project_with_a_blank_name_gets_a_placeholder(store: Store) -> None:
    assert store.create_project("  ").name == "Untitled project"


def test_ids_are_unique_and_prefixed(store: Store) -> None:
    ids = {store.new_id("msg") for _ in range(50)}

    assert len(ids) == 50
    assert all(generated.startswith("msg_") for generated in ids)


# ----------------------------------------------------------------- connections


def test_there_is_no_active_connection_on_a_fresh_store(store: Store) -> None:
    assert store.active_connection() is None


def test_an_active_connection_id_pointing_nowhere_resolves_to_none(store: Store) -> None:
    store.settings.active_connection_id = "conn_deleted"

    assert store.active_connection() is None
