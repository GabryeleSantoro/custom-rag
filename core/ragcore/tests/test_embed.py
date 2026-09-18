"""The embed client's contract: batching, dimension, normalisation."""

from __future__ import annotations

import math

import pytest
from ragcore.models.embed import EMBED_DIM, EmbedClient
from ragcore.models.fakes import FakeEmbedClient


async def test_fake_client_is_deterministic_and_the_right_shape() -> None:
    client = FakeEmbedClient()
    first = await client.embed(["reranking reorders candidates"])
    second = await client.embed(["reranking reorders candidates"])

    assert len(first) == 1
    assert len(first[0]) == EMBED_DIM
    assert first == second


async def test_fake_client_returns_unit_vectors() -> None:
    [vector] = await FakeEmbedClient().embed(["chunking strategies"])

    assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-6)


async def test_fake_client_separates_different_texts() -> None:
    a, b = await FakeEmbedClient().embed(["hybrid search", "optical character recognition"])

    assert sum(x * y for x, y in zip(a, b, strict=True)) < 0.9


async def test_batches_are_split(monkeypatch: pytest.MonkeyPatch) -> None:
    client = EmbedClient("http://127.0.0.1:8770")
    seen: list[int] = []

    async def fake_post(texts: list[str]) -> list[list[float]]:
        seen.append(len(texts))
        return [[0.0] * EMBED_DIM for _ in texts]

    monkeypatch.setattr(client, "_post", fake_post)
    await client.embed([f"text {i}" for i in range(70)], batch_size=32)

    assert seen == [32, 32, 6]


async def test_fake_client_handles_empty_texts() -> None:
    result = await FakeEmbedClient().embed([])

    assert result == []


async def test_embed_client_handles_empty_texts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = EmbedClient("http://127.0.0.1:8770")
    called = False

    async def fake_post(texts: list[str]) -> list[list[float]]:
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(client, "_post", fake_post)
    result = await client.embed([])

    assert result == []
    assert not called


@pytest.mark.requires_models
async def test_live_server_returns_1024_dimensions() -> None:
    client = EmbedClient("http://127.0.0.1:8770")
    try:
        [vector] = await client.embed(["reranking reorders candidates"])
    finally:
        await client.aclose()

    assert len(vector) == EMBED_DIM
    assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-3)
