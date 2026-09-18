"""Naive fixed-size chunking with a contextual header.

Word-based on purpose: the Qwen3 tokenizer and parent/child splitting land with
real chunking, and pulling `tokenizers` in now would be a dependency we then
have to justify twice.
"""

from __future__ import annotations

from ragcore.ingest.chunk import chunk_document
from ragcore.ingest.parse import ParsedDoc, ParsedPage


def doc_with(words: int, pages: int = 1) -> ParsedDoc:
    return ParsedDoc(
        title="Reranking",
        pages=[
            ParsedPage(
                page=i + 1,
                section_path=f"Reranking > Part {i + 1}",
                text=" ".join(f"word{j}" for j in range(words)),
            )
            for i in range(pages)
        ],
    )


def test_short_page_becomes_one_chunk() -> None:
    chunks = chunk_document("doc_1", doc_with(50))

    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].n_tokens == 50


def test_long_page_splits_with_overlap() -> None:
    chunks = chunk_document("doc_1", doc_with(600), target_words=256, overlap_words=30)

    assert len(chunks) == 3
    first_tail = chunks[0].text.split()[-30:]
    second_head = chunks[1].text.split()[:30]
    assert first_tail == second_head


def test_embed_text_carries_the_contextual_header() -> None:
    [chunk] = chunk_document("doc_1", doc_with(20))

    assert chunk.embed_text.startswith("Reranking > Part 1\n")
    assert chunk.embed_text.endswith(chunk.text)


def test_chunks_never_span_pages() -> None:
    chunks = chunk_document("doc_1", doc_with(300, pages=2))

    for chunk in chunks:
        assert chunk.page_start == chunk.page_end
    assert {c.page_start for c in chunks} == {1, 2}


def test_chunk_ids_are_unique_and_prefixed_by_document() -> None:
    chunks = chunk_document("doc_1", doc_with(600))
    ids = [c.chunk_id for c in chunks]

    assert len(set(ids)) == len(ids)
    assert all(cid.startswith("doc_1:") for cid in ids)


def test_char_offsets_point_back_into_the_page() -> None:
    parsed = doc_with(600)
    chunks = chunk_document("doc_1", parsed)
    page_text = parsed.pages[0].text

    for chunk in chunks:
        assert page_text[chunk.char_start : chunk.char_end] == chunk.text


def test_empty_pages_are_skipped() -> None:
    parsed = ParsedDoc(title="Empty", pages=[ParsedPage(1, "Empty", "   ")])

    assert chunk_document("doc_1", parsed) == []
