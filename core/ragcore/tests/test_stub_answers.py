"""The scripted answer: built from the retrieved passages, citing them."""

from __future__ import annotations

from ragcore.api.schemas import RetrievedChunk
from ragcore.stub import answers


def test_the_scripted_answer_cites_the_passage_it_leads_with() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="c1", doc_id="rerank", doc_title="Reranking", page_start=2, page_end=2,
            text="Reranking reorders candidates. It runs after fusion.",
        ),
        RetrievedChunk(
            chunk_id="c2", doc_id="fusion", doc_title="Fusion", page_start=1, page_end=1,
            text="Reciprocal rank fusion merges two rankings.",
        ),
    ]

    text = answers.compose_answer("Why rerank?", chunks)

    assert text.startswith("Reranking reorders candidates. [rerank:2]")
    assert "[fusion:1]" in text
    assert "Drawn from 2 documents" in text


def test_the_scripted_answer_says_so_when_nothing_was_retrieved() -> None:
    assert answers.compose_answer("Why rerank?", []) == answers.NOT_FOUND


def test_a_single_document_answer_skips_the_provenance_line() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}", doc_id="rerank", doc_title="Reranking", page_start=i, page_end=i,
            text=f"Sentence {i}. More text.",
        )
        for i in (1, 2)
    ]

    assert "Drawn from" not in answers.compose_answer("Why rerank?", chunks)


def test_a_very_long_sentence_is_elided_rather_than_dumped() -> None:
    chunk = RetrievedChunk(
        chunk_id="c1", doc_id="long", doc_title="Long", page_start=1, page_end=1,
        text="word " * 200 + ".",
    )

    lead = answers.compose_answer("q", [chunk]).split(" [long:1]")[0]

    assert len(lead) <= 240
    assert lead.endswith("…")
