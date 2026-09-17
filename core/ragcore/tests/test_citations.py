"""Citation parsing is contract-critical: a marker nothing retrieved must never render."""

from __future__ import annotations

from ragcore.api.schemas import RetrievedChunk
from ragcore.citations import extract_citations, parse_directives


def chunk(doc_id: str, page: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"c_{doc_id}_{page}",
        doc_id=doc_id,
        doc_title=doc_id.title(),
        page_start=page,
        page_end=page,
        text="Reranking reorders candidates with a cross-encoder.",
        rerank_score=0.9,
    )


def test_marker_backed_by_a_retrieved_chunk_is_kept() -> None:
    citations, dropped, grounding = extract_citations(
        "Rerankers reorder results [reranking:1] and improve precision [reranking:2].",
        [chunk("reranking", 1), chunk("reranking", 2)],
    )

    assert [c.marker for c in citations] == ["[reranking:1]", "[reranking:2]"]
    assert dropped == 0
    assert grounding == "ok"


def test_marker_for_an_unretrieved_document_is_dropped() -> None:
    citations, dropped, grounding = extract_citations(
        "As shown in [invented:7], the answer is yes.", [chunk("reranking", 1)]
    )

    assert citations == []
    assert dropped == 1
    assert grounding == "none"


def test_repeated_marker_is_emitted_once() -> None:
    citations, dropped, _ = extract_citations(
        "First [reranking:1]. Again [reranking:1].", [chunk("reranking", 1)]
    )

    assert len(citations) == 1
    assert dropped == 0


def test_directives_are_stripped_from_the_question() -> None:
    question, directives = parse_directives("!slow what is reranking?")

    assert question == "what is reranking?"
    assert "slow" in directives
