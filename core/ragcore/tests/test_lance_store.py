"""LanceDB tables: write, search, delete, and the index_meta guard."""

from __future__ import annotations

from pathlib import Path

from ragcore.models.fakes import FakeEmbedClient
from ragcore.store.lance import VectorStore


async def rows_for(texts: list[str], doc_id: str = "doc_1") -> list[dict]:
    vectors = await FakeEmbedClient().embed(texts)
    return [
        {
            "chunk_id": f"chk_{i}",
            "doc_id": doc_id,
            "doc_title": "Reranking",
            "page_start": 1,
            "page_end": 1,
            "section_path": "Reranking > Overview",
            "text": text,
            "embed_text": text,
            "n_tokens": len(text.split()),
            "vector": vector,
        }
        for i, (text, vector) in enumerate(zip(texts, vectors, strict=True))
    ]


async def test_chunks_are_written_and_counted(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "index")
    await_rows = await rows_for(["reranking reorders", "chunking splits text"])
    store.add_chunks(await_rows)

    assert store.chunk_count() == 2


async def test_dense_search_finds_the_matching_chunk(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "index")
    texts = ["reranking reorders candidates", "optical character recognition of scans"]
    store.add_chunks(await rows_for(texts))
    [query_vector] = await FakeEmbedClient().embed([texts[0]])

    hits = store.dense(query_vector, k=2)

    assert hits[0][0] == "chk_0"
    assert hits[0][1] > hits[1][1]


async def test_deleting_a_document_removes_its_chunks(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "index")
    store.add_chunks(await rows_for(["a reranking passage"], doc_id="doc_1"))
    store.add_chunks(await rows_for(["a chunking passage"], doc_id="doc_2"))

    store.delete_by_doc(["doc_1"])

    assert store.chunk_count() == 1
    assert store.get(["chk_0"])[0]["doc_id"] == "doc_2"


def test_meta_round_trips(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "index")
    store.write_meta({"embed_model": "Qwen3-Embedding-0.6B-Q8_0", "embed_dim": "1024"})

    assert store.read_meta()["embed_dim"] == "1024"


def test_meta_survives_a_reopen(tmp_path: Path) -> None:
    root = tmp_path / "index"
    VectorStore(root).write_meta({"schema_version": "1"})

    assert VectorStore(root).read_meta()["schema_version"] == "1"


async def test_deleting_an_empty_list_deletes_nothing(tmp_path: Path) -> None:
    """An empty `IN ()` predicate must never be read as "delete everything"."""
    store = VectorStore(tmp_path / "index")
    store.add_chunks(await rows_for(["a reranking passage"]))

    store.delete_by_doc([])

    assert store.chunk_count() == 1


async def test_adding_no_rows_is_a_no_op(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "index")

    store.add_chunks([])

    assert store.chunk_count() == 0


async def test_get_returns_rows_in_the_order_they_were_asked_for(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "index")
    store.add_chunks(await rows_for(["first passage", "second passage", "third passage"]))

    rows = store.get(["chk_2", "chk_0"])

    assert [row["chunk_id"] for row in rows] == ["chk_2", "chk_0"]


async def test_get_silently_drops_ids_that_are_not_there(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "index")
    store.add_chunks(await rows_for(["a passage"]))

    rows = store.get(["chk_0", "chk_missing"])

    assert [row["chunk_id"] for row in rows] == ["chk_0"]


async def test_getting_nothing_returns_nothing(tmp_path: Path) -> None:
    assert VectorStore(tmp_path / "index").get([]) == []


async def test_a_chunk_round_trips_with_its_text_and_offsets(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "index")
    store.add_chunks(await rows_for(["reranking reorders candidates"]))

    [row] = store.get(["chk_0"])

    assert row["text"] == "reranking reorders candidates"
    assert row["section_path"] == "Reranking > Overview"
    assert (row["page_start"], row["page_end"]) == (1, 1)
    assert len(row["vector"]) == 1024


async def test_dense_search_on_an_empty_table_finds_nothing(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "index")
    [vector] = await FakeEmbedClient().embed(["anything"])

    assert store.dense(vector, k=6) == []


async def test_asking_for_more_neighbours_than_exist_is_not_an_error(
    tmp_path: Path,
) -> None:
    store = VectorStore(tmp_path / "index")
    store.add_chunks(await rows_for(["only passage"]))
    [vector] = await FakeEmbedClient().embed(["only passage"])

    assert len(store.dense(vector, k=50)) == 1


async def test_full_text_search_is_not_built_yet_and_says_so(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(NotImplementedError, match="Task 12"):
        VectorStore(tmp_path / "index").fts("reranking", 6)


async def test_writing_the_same_meta_key_twice_replaces_it(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "index")
    store.write_meta({"embed_model": "old"})

    store.write_meta({"embed_model": "new"})

    assert store.read_meta() == {"embed_model": "new"}


async def test_chunks_survive_a_reopen(tmp_path: Path) -> None:
    root = tmp_path / "index"
    VectorStore(root).add_chunks(await rows_for(["a passage"]))

    assert VectorStore(root).chunk_count() == 1
