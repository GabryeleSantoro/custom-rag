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
