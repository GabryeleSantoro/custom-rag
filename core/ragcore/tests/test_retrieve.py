"""Retrieval pipeline over a fake embedder: ranks, filters, budget, latency."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ragcore.api.schemas import QueryFilters, RetrievalSettings, RetrievedChunk
from ragcore.models.fakes import FakeEmbedClient
from ragcore.retrieve.hybrid import HybridRetriever
from ragcore.retrieve.pack import pack
from ragcore.store.lance import VectorStore

TEXTS = {
    "chk_rerank": "Cross encoders rerank candidate passages by scoring pairs.",
    "chk_chunk": "Chunk size trades precision against context completeness.",
    "chk_ocr": "Optical character recognition extracts text from scanned images.",
}

DOC_META = {
    cid.replace("chk_", "doc_"): {
        "source_id": "src_1",
        "ext": ".md",
        "lang": None,
        "mtime": datetime(2026, 1, 1, tzinfo=UTC),
    }
    for cid in TEXTS
}


class RecordingEmbedClient(FakeEmbedClient):
    def __init__(self) -> None:
        self.query_flags: list[bool] = []

    async def embed(self, texts, *, batch_size=32, query=False):
        self.query_flags.append(query)
        return await super().embed(texts, batch_size=batch_size, query=query)


async def build(tmp_path: Path, embedder: FakeEmbedClient | None = None) -> HybridRetriever:
    vectors = VectorStore(tmp_path / "index")
    embedder = embedder or FakeEmbedClient()
    embedded = await embedder.embed(list(TEXTS.values()))
    vectors.add_chunks(
        [
            {
                "chunk_id": cid,
                "doc_id": cid.replace("chk_", "doc_"),
                "doc_title": cid,
                "page_start": 1,
                "page_end": 1,
                "section_path": "Doc > Section",
                "text": text,
                "embed_text": text,
                "n_tokens": len(text.split()),
                "vector": vector,
            }
            for (cid, text), vector in zip(TEXTS.items(), embedded, strict=True)
        ]
    )
    return HybridRetriever(vectors, embedder)


async def test_dense_search_ranks_the_exact_text_first(tmp_path: Path) -> None:
    retriever = await build(tmp_path)

    chunks, latency, candidates = await retriever.asearch(
        TEXTS["chk_rerank"],
        settings=RetrievalSettings(),
        filters=QueryFilters(),
        doc_meta=DOC_META,
    )

    assert chunks[0].chunk_id == "chk_rerank"
    assert candidates >= 1
    assert latency.embed_ms > 0
    assert latency.dense_ms > 0
    assert latency.total_ms >= latency.embed_ms + latency.dense_ms


async def test_the_question_is_embedded_with_the_query_prompt(tmp_path: Path) -> None:
    embedder = RecordingEmbedClient()
    retriever = await build(tmp_path, embedder)

    await retriever.asearch(
        "scanned images", settings=RetrievalSettings(), filters=QueryFilters(), doc_meta=DOC_META
    )

    assert embedder.query_flags[-1] is True


async def test_doc_id_filter_restricts_results(tmp_path: Path) -> None:
    retriever = await build(tmp_path)

    chunks, _, _ = await retriever.asearch(
        TEXTS["chk_rerank"],
        settings=RetrievalSettings(),
        filters=QueryFilters(doc_ids=["doc_ocr"]),
        doc_meta=DOC_META,
    )

    assert {c.doc_id for c in chunks} == {"doc_ocr"}


async def test_a_filter_applies_before_the_dense_cut_not_after(tmp_path: Path) -> None:
    retriever = await build(tmp_path)

    chunks, _, _ = await retriever.asearch(
        TEXTS["chk_rerank"],
        settings=RetrievalSettings(dense_top_k=1),
        filters=QueryFilters(doc_ids=["doc_ocr"]),
        doc_meta=DOC_META,
    )

    assert [c.chunk_id for c in chunks] == ["chk_ocr"]


async def test_an_empty_source_filter_allows_nothing(tmp_path: Path) -> None:
    retriever = await build(tmp_path)

    chunks, _, candidates = await retriever.asearch(
        TEXTS["chk_rerank"],
        settings=RetrievalSettings(),
        filters=QueryFilters(source_ids=[]),
        doc_meta=DOC_META,
    )

    assert chunks == []
    assert candidates == 0


async def test_chunks_of_a_document_the_library_no_longer_lists_are_dropped(
    tmp_path: Path,
) -> None:
    retriever = await build(tmp_path)
    listed = {k: v for k, v in DOC_META.items() if k != "doc_rerank"}

    chunks, _, _ = await retriever.asearch(
        TEXTS["chk_rerank"], settings=RetrievalSettings(), filters=QueryFilters(), doc_meta=listed
    )

    assert "doc_rerank" not in {c.doc_id for c in chunks}


async def test_top_k_is_honoured(tmp_path: Path) -> None:
    retriever = await build(tmp_path)

    chunks, _, _ = await retriever.asearch(
        "text",
        settings=RetrievalSettings(top_k=2, min_score=0.0),
        filters=QueryFilters(),
        doc_meta=DOC_META,
    )

    assert len(chunks) <= 2


async def test_empty_query_returns_nothing(tmp_path: Path) -> None:
    retriever = await build(tmp_path)

    chunks, _, candidates = await retriever.asearch(
        "   ", settings=RetrievalSettings(), filters=QueryFilters(), doc_meta=DOC_META
    )

    assert chunks == []
    assert candidates == 0


def a_chunk(chunk_id: str, tokens: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc_1",
        doc_title="Doc",
        page_start=1,
        page_end=1,
        text=" ".join(["word"] * tokens),
        rerank_score=score,
    )


def test_packing_stops_at_the_budget() -> None:
    packed = pack([a_chunk("a", 100, 0.9), a_chunk("b", 100, 0.8), a_chunk("c", 100, 0.7)], 250)

    assert [c.chunk_id for c in packed] == ["a", "b"]


def test_packing_keeps_score_order() -> None:
    packed = pack([a_chunk("low", 10, 0.1), a_chunk("high", 10, 0.9)], 1000)

    assert [c.chunk_id for c in packed] == ["high", "low"]


def test_packing_never_returns_nothing_when_a_chunk_fits_alone() -> None:
    packed = pack([a_chunk("huge", 5000, 0.9)], 100)

    assert [c.chunk_id for c in packed] == ["huge"]
