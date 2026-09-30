"""Filling the context budget in score order.

Parent mapping belongs here once real chunking produces parents. Today the
children are what the LLM sees, so packing is a greedy fill that always keeps
at least the best chunk, even when it alone exceeds the budget — an answer from
one over-long passage beats no answer at all.
"""

from __future__ import annotations

from ragcore.api.schemas import RetrievedChunk


def _tokens(chunk: RetrievedChunk) -> int:
    return len(chunk.text.split())


def pack(chunks: list[RetrievedChunk], budget_tokens: int) -> list[RetrievedChunk]:
    ordered = sorted(chunks, key=lambda c: -c.rerank_score)
    packed: list[RetrievedChunk] = []
    used = 0
    for chunk in ordered:
        cost = _tokens(chunk)
        if packed and used + cost > budget_tokens:
            continue
        packed.append(chunk)
        used += cost
    return packed
