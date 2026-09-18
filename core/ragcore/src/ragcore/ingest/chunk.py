"""Splitting pages into retrievable units.

Naive by design: fixed word windows with overlap, one page at a time, never
crossing a page boundary because a citation names exactly one page. Heading-
aware parent/child chunking with real token counts replaces this wholesale.

`embed_text` prepends the section path. The embedder sees where a passage sits;
the LLM and the reader see only `text`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ragcore.ingest.parse import ParsedDoc


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    page_start: int
    page_end: int
    section_path: str
    text: str
    embed_text: str
    n_tokens: int
    char_start: int
    char_end: int


def _windows(count: int, target: int, overlap: int) -> list[tuple[int, int]]:
    """Word-index spans covering ``[0, count)``.

    A single span if the page already fits in ``target`` words. Otherwise
    fixed-size windows advancing by ``target - overlap`` words (at least 1,
    so a degenerate overlap can never stall the loop), with the final window
    pulled back to end exactly at ``count`` so no trailing words are dropped.
    """
    if count <= target:
        return [(0, count)]
    step = max(target - overlap, 1)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < count:
        spans.append((start, min(start + target, count)))
        if start + target >= count:
            break
        start += step
    return spans


def chunk_document(
    doc_id: str,
    parsed: ParsedDoc,
    *,
    target_words: int = 256,
    overlap_words: int = 30,
) -> list[Chunk]:
    """Split every page of ``parsed`` into word-window ``Chunk``s.

    ``n_tokens`` is a word count, not a real tokenizer count — see the module
    docstring for why. Chunks never span pages: a citation names exactly one
    page, so ``page_start == page_end`` for every chunk produced here.
    """
    chunks: list[Chunk] = []
    for page in parsed.pages:
        text = page.text.strip()
        if not text:
            continue
        words = text.split()

        # Recover each word's (start, end) offset by walking the original
        # string left to right. Searching from `cursor` (the end of the
        # previous word) guarantees the match found is that word's own
        # token, not an earlier occurrence of the same substring, because
        # only whitespace can separate `cursor` from the next real token.
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for word in words:
            start = text.index(word, cursor)
            offsets.append((start, start + len(word)))
            cursor = start + len(word)

        for lo, hi in _windows(len(words), target_words, overlap_words):
            char_start, char_end = offsets[lo][0], offsets[hi - 1][1]
            body = text[char_start:char_end]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:{page.page}:{lo}",
                    doc_id=doc_id,
                    doc_title=parsed.title,
                    page_start=page.page,
                    page_end=page.page,
                    section_path=page.section_path,
                    text=body,
                    embed_text=f"{page.section_path}\n{body}",
                    n_tokens=hi - lo,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
    return chunks
