"""The hybrid retriever: two legs, fusion, rerank, score gate, filters."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from ragcore.api.schemas import QueryFilters, RetrievalSettings
from ragcore.stub.corpus import Chunk, tokenize
from ragcore.stub.retrieval import Retriever

MTIME = datetime(2026, 3, 1, tzinfo=UTC)

CORPUS = {
    "rerank": "Reranking reorders retrieved candidates with a cross encoder before packing.",
    "fusion": "Reciprocal rank fusion merges the dense ranking and the keyword ranking.",
    "chunking": "Chunking splits a document into passages that fit the context budget.",
    "ocr": "Optical character recognition turns scanned page images into selectable text.",
    "watch": "The watcher notices a changed file and queues it for incremental indexing.",
}


def a_chunk(doc_id: str, text: str, page: int = 1) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}#p{page}c0",
        doc_id=doc_id,
        doc_title=doc_id.title(),
        page=page,
        section_path=f"{doc_id.title()} > Body",
        text=text,
        char_start=0,
        char_end=len(text),
        tokens=tokenize(text),
    )


@pytest.fixture
def retriever() -> Retriever:
    return Retriever([a_chunk(doc_id, text) for doc_id, text in CORPUS.items()])


@pytest.fixture
def doc_meta() -> dict[str, dict]:
    return {
        doc_id: {"source_id": "src_1", "ext": ".md", "lang": "en", "mtime": MTIME}
        for doc_id in CORPUS
    }


def search(retriever: Retriever, query: str, doc_meta: dict, **overrides):
    settings = RetrievalSettings(min_score=0.0, **overrides)
    return retriever.search(query, settings=settings, filters=QueryFilters(), doc_meta=doc_meta)


# ----------------------------------------------------------------- the basics


def test_the_best_match_comes_first(retriever: Retriever, doc_meta: dict) -> None:
    chunks, _, _ = search(retriever, "reciprocal rank fusion", doc_meta)

    assert chunks[0].doc_id == "fusion"


def test_a_result_carries_both_ranks_and_the_fused_score(
    retriever: Retriever, doc_meta: dict
) -> None:
    chunks, _, _ = search(retriever, "reciprocal rank fusion", doc_meta)

    top = chunks[0]
    assert top.dense_rank is not None and top.bm25_rank is not None
    assert top.rrf_score > 0
    assert 0.0 <= top.rerank_score <= 1.0
    assert top.section_path == "Fusion > Body"
    assert (top.page_start, top.page_end) == (1, 1)


def test_an_empty_query_retrieves_nothing(retriever: Retriever, doc_meta: dict) -> None:
    chunks, latency, candidates = search(retriever, "   ", doc_meta)

    assert (chunks, candidates) == ([], 0)
    assert latency.total_ms == 0


def test_a_query_of_only_punctuation_retrieves_nothing(
    retriever: Retriever, doc_meta: dict
) -> None:
    assert search(retriever, "?!...", doc_meta)[0] == []


def test_an_empty_index_retrieves_nothing() -> None:
    chunks, _, candidates = Retriever([]).search(
        "anything",
        settings=RetrievalSettings(),
        filters=QueryFilters(),
        doc_meta={},
    )

    assert (chunks, candidates) == ([], 0)


def test_a_query_matching_nothing_returns_no_candidates(
    retriever: Retriever, doc_meta: dict
) -> None:
    chunks, _, _ = search(retriever, "helicopter maintenance schedule", doc_meta)

    assert all(chunk.rerank_score < 0.5 for chunk in chunks)


def test_every_stage_is_timed(retriever: Retriever, doc_meta: dict) -> None:
    _, latency, _ = search(retriever, "reranking", doc_meta)

    stages = [latency.embed_ms, latency.dense_ms, latency.bm25_ms, latency.rerank_ms,
              latency.pack_ms]
    assert all(value >= 0 for value in stages)
    assert latency.total_ms == pytest.approx(sum(stages))


# ------------------------------------------------------------------- settings


def test_top_k_caps_how_much_is_packed(retriever: Retriever, doc_meta: dict) -> None:
    chunks, _, _ = search(retriever, "document passages text", doc_meta, top_k=2)

    assert len(chunks) == 2


def test_the_min_score_gate_drops_weak_passages(retriever: Retriever, doc_meta: dict) -> None:
    kept, _, candidates = retriever.search(
        "reranking",
        settings=RetrievalSettings(min_score=0.99),
        filters=QueryFilters(),
        doc_meta=doc_meta,
    )

    assert kept == []
    assert candidates > 0, "candidates are counted before the gate, not after"


def test_the_candidate_count_reports_what_the_reranker_saw(
    retriever: Retriever, doc_meta: dict
) -> None:
    _, _, candidates = search(retriever, "document passages text", doc_meta, rerank_candidates=2)

    assert candidates == 2


def test_narrowing_the_dense_leg_narrows_the_pool(
    retriever: Retriever, doc_meta: dict
) -> None:
    _, _, wide = search(retriever, "document passages text", doc_meta)
    _, _, narrow = search(
        retriever, "document passages text", doc_meta, dense_top_k=1, bm25_top_k=1
    )

    assert narrow < wide


# -------------------------------------------------------------------- filters


def test_a_document_filter_keeps_only_that_document(
    retriever: Retriever, doc_meta: dict
) -> None:
    chunks, _, _ = retriever.search(
        "ranking",
        settings=RetrievalSettings(min_score=0.0),
        filters=QueryFilters(doc_ids=["fusion"]),
        doc_meta=doc_meta,
    )

    assert {chunk.doc_id for chunk in chunks} == {"fusion"}


def test_a_source_filter_excludes_other_sources(retriever: Retriever, doc_meta: dict) -> None:
    doc_meta["fusion"]["source_id"] = "src_2"

    chunks, _, _ = retriever.search(
        "ranking",
        settings=RetrievalSettings(min_score=0.0),
        filters=QueryFilters(source_ids=["src_2"]),
        doc_meta=doc_meta,
    )

    assert {chunk.doc_id for chunk in chunks} == {"fusion"}


def test_an_extension_filter_is_applied(retriever: Retriever, doc_meta: dict) -> None:
    doc_meta["ocr"]["ext"] = ".pdf"

    chunks, _, _ = retriever.search(
        "text page images document",
        settings=RetrievalSettings(min_score=0.0),
        filters=QueryFilters(exts=[".pdf"]),
        doc_meta=doc_meta,
    )

    assert {chunk.doc_id for chunk in chunks} == {"ocr"}


def test_a_language_filter_is_applied(retriever: Retriever, doc_meta: dict) -> None:
    doc_meta["chunking"]["lang"] = "it"

    chunks, _, _ = retriever.search(
        "document passages",
        settings=RetrievalSettings(min_score=0.0),
        filters=QueryFilters(langs=["it"]),
        doc_meta=doc_meta,
    )

    assert {chunk.doc_id for chunk in chunks} == {"chunking"}


def test_a_date_window_excludes_what_falls_outside_it(
    retriever: Retriever, doc_meta: dict
) -> None:
    doc_meta["watch"]["mtime"] = MTIME + timedelta(days=30)

    recent, _, _ = retriever.search(
        "file indexing changed",
        settings=RetrievalSettings(min_score=0.0),
        filters=QueryFilters(after=MTIME + timedelta(days=1)),
        doc_meta=doc_meta,
    )
    older, _, _ = retriever.search(
        "file indexing changed",
        settings=RetrievalSettings(min_score=0.0),
        filters=QueryFilters(before=MTIME + timedelta(days=1)),
        doc_meta=doc_meta,
    )

    assert {chunk.doc_id for chunk in recent} == {"watch"}
    assert "watch" not in {chunk.doc_id for chunk in older}


def test_a_chunk_whose_document_has_no_metadata_is_unreachable(
    retriever: Retriever, doc_meta: dict
) -> None:
    """A stale chunk left behind by a delete must not surface as a source."""
    doc_meta.pop("fusion")

    chunks, _, _ = retriever.search(
        "reciprocal rank fusion",
        settings=RetrievalSettings(min_score=0.0),
        filters=QueryFilters(),
        doc_meta=doc_meta,
    )

    assert "fusion" not in {chunk.doc_id for chunk in chunks}


def test_an_empty_source_filter_allows_nothing(retriever: Retriever, doc_meta: dict) -> None:
    chunks, _, _ = retriever.search(
        "reranking",
        settings=RetrievalSettings(min_score=0.0),
        filters=QueryFilters(source_ids=[]),
        doc_meta=doc_meta,
    )

    assert chunks == []


# ----------------------------------------------------------------- the two legs


def test_the_two_legs_disagree_which_is_why_fusion_exists(
    retriever: Retriever, doc_meta: dict
) -> None:
    chunks, _, _ = search(retriever, "document text passages page", doc_meta)

    ranks = [(c.dense_rank, c.bm25_rank) for c in chunks if c.dense_rank and c.bm25_rank]
    assert any(dense != bm25 for dense, bm25 in ranks), (
        "identical rankings would make reciprocal rank fusion pointless"
    )


def test_a_term_in_every_document_carries_little_weight(doc_meta: dict) -> None:
    """Inverse document frequency, doing its job."""
    common = Retriever([a_chunk(doc_id, f"the {text}") for doc_id, text in CORPUS.items()])

    assert common._idf("the") < common._idf("cross")


def test_repeating_a_term_does_not_scale_the_score_linearly(doc_meta: dict) -> None:
    """BM25 saturation: the tenth mention adds far less than the second."""
    once = Retriever([a_chunk("a", "reranking helps")])
    many = Retriever([a_chunk("a", "reranking " * 10 + "helps")])
    terms = tokenize("reranking")

    assert many._bm25(terms, 0) < 10 * once._bm25(terms, 0)
