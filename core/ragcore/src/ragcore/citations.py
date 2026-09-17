"""Citation marker parsing and dev directives.

Backend-agnostic: both the stub and the real backend produce answers containing
`[doc_id:page]` markers, and both must drop any marker that no retrieved passage
backs. Lives outside `stub/` because the API layer imports it directly.
"""

from __future__ import annotations

import re

from ragcore.api.schemas import Citation, Grounding, RetrievedChunk

MARKER = re.compile(r"\[([A-Za-z0-9_.\-]+):(\d+)\]")


def extract_citations(
    text: str, chunks: list[RetrievedChunk]
) -> tuple[list[Citation], int, Grounding]:
    """Parse markers, keep the ones backed by a retrieved passage, drop the rest."""
    by_key = {(c.doc_id, c.page_start): c for c in chunks}
    citations: list[Citation] = []
    seen: set[str] = set()
    dropped = 0

    for match in MARKER.finditer(text):
        doc_id, page_raw = match.group(1), int(match.group(2))
        chunk = by_key.get((doc_id, page_raw))
        if chunk is None:
            dropped += 1
            continue
        if match.group(0) in seen:
            continue
        seen.add(match.group(0))
        citations.append(
            Citation(
                marker=match.group(0),
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                page=chunk.page_start,
                chunk_id=chunk.chunk_id,
                section_path=chunk.section_path,
                snippet=chunk.text[:400],
                score=chunk.rerank_score,
            )
        )

    if not citations:
        grounding: Grounding = "none"
    elif dropped or len(citations) < 2:
        grounding = "low"
    else:
        grounding = "ok"
    return citations, dropped, grounding


def parse_directives(question: str) -> tuple[str, set[str]]:
    directives: set[str] = set()
    text = question.strip()
    while text.startswith("!"):
        token, _, rest = text.partition(" ")
        directives.add(token[1:].lower())
        text = rest.strip()
    return text or question.strip(), directives
