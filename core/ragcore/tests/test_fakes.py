"""The deterministic stand-ins that let the real pipeline be tested offline."""

from __future__ import annotations

from ragcore.models.fakes import FakeRerankClient

DOCS = [
    "chunking splits a document into passages",
    "reranking reorders retrieved candidates with a cross encoder",
    "the watcher queues a changed file",
]


async def test_the_reranker_puts_the_best_overlap_first() -> None:
    ranked = await FakeRerankClient().rerank("reranking candidates", DOCS, top_n=3)

    assert ranked[0][0] == 1


async def test_the_reranker_returns_index_and_score_pairs() -> None:
    ranked = await FakeRerankClient().rerank("chunking passages", DOCS, top_n=3)

    assert sorted(index for index, _ in ranked) == [0, 1, 2]
    assert all(0.0 <= score <= 1.0 for _, score in ranked)


async def test_the_reranker_honours_top_n() -> None:
    assert len(await FakeRerankClient().rerank("chunking", DOCS, top_n=1)) == 1


async def test_scores_come_back_in_descending_order() -> None:
    scores = [score for _, score in await FakeRerankClient().rerank("chunking", DOCS, top_n=3)]

    assert scores == sorted(scores, reverse=True)


async def test_a_query_matching_nothing_scores_everything_zero() -> None:
    ranked = await FakeRerankClient().rerank("helicopter", DOCS, top_n=3)

    assert {score for _, score in ranked} == {0.0}


async def test_an_empty_query_does_not_divide_by_zero() -> None:
    ranked = await FakeRerankClient().rerank("", DOCS, top_n=3)

    assert len(ranked) == 3


async def test_reranking_nothing_returns_nothing() -> None:
    assert await FakeRerankClient().rerank("chunking", [], top_n=6) == []
