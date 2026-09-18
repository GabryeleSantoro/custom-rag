"""Deterministic stand-ins so the real backend is testable without llama-server.

Vectors are hash-derived, so the same text always lands in the same place and
different texts land apart. They carry no semantics — these are for wiring
tests, never for quality measurement.
"""

from __future__ import annotations

import hashlib

import numpy as np

from ragcore.models.embed import EMBED_DIM


def _vector(text: str) -> list[float]:
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return (vector / np.linalg.norm(vector)).tolist()


class FakeEmbedClient:
    dim = EMBED_DIM

    async def embed(self, texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
        return [_vector(text) for text in texts]

    async def aclose(self) -> None:
        return None


class FakeRerankClient:
    """Scores by literal token overlap: crude, but it reorders, which is what tests check."""

    async def rerank(
        self, query: str, documents: list[str], *, top_n: int
    ) -> list[tuple[int, float]]:
        terms = set(query.lower().split())
        scored = [
            (i, len(terms & set(doc.lower().split())) / (len(terms) or 1))
            for i, doc in enumerate(documents)
        ]
        scored.sort(key=lambda pair: -pair[1])
        return scored[:top_n]

    async def aclose(self) -> None:
        return None
