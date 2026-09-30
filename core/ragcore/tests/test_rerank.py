"""The rerank client: llama-server's scores pass through, best first, truncated."""

from __future__ import annotations

import pytest
from ragcore.models.rerank import RerankClient


async def test_client_sorts_by_score_and_truncates(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RerankClient("http://127.0.0.1:8771")

    async def fake_post(query: str, documents: list[str], top_n: int) -> list[dict]:
        return [
            {"index": 0, "relevance_score": 0.01},
            {"index": 1, "relevance_score": 0.98},
            {"index": 2, "relevance_score": 0.4},
        ]

    monkeypatch.setattr(client, "_post", fake_post)

    assert await client.rerank("q", ["a", "b", "c"], top_n=2) == [(1, 0.98), (2, 0.4)]


@pytest.mark.requires_models
async def test_live_reranker_scores_the_relevant_passage_as_a_probability() -> None:
    client = RerankClient("http://127.0.0.1:8771")
    try:
        ranked = await client.rerank(
            "what do cross encoders do",
            [
                "Optical character recognition extracts text from scanned images.",
                "Cross encoders score a query and passage together and reorder candidates.",
            ],
            top_n=2,
        )
    finally:
        await client.aclose()

    assert ranked[0][0] == 1
    assert ranked[0][1] > 0.5 > ranked[1][1] >= 0.0
