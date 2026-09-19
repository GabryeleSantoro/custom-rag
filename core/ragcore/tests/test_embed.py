"""The embed client's contract: batching, dimension, normalisation."""

from __future__ import annotations

import json
import math

import httpx
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


async def test_embeddings_are_reordered_by_the_index_the_server_returned() -> None:
    """llama-server may answer out of order; the caller's order is the contract."""
    client = EmbedClient("http://127.0.0.1:8770", normalize=False)
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "embedding": [2.0] * EMBED_DIM},
                        {"index": 0, "embedding": [1.0] * EMBED_DIM},
                    ]
                },
            )
        )
    )

    first, second = await client.embed(["a", "b"])

    assert first[0] == 1.0
    assert second[0] == 2.0


async def test_the_request_names_the_model_and_carries_every_text() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        assert request.url.path == "/v1/embeddings"
        return httpx.Response(
            200, json={"data": [{"index": 0, "embedding": [1.0] * EMBED_DIM}]}
        )

    client = EmbedClient("http://127.0.0.1:8770/", model="qwen3-embedding")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    await client.embed(["reranking"])

    assert seen == {"input": ["reranking"], "model": "qwen3-embedding"}


async def test_vectors_come_back_normalised() -> None:
    client = EmbedClient("http://127.0.0.1:8770")
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"data": [{"index": 0, "embedding": [3.0] + [0.0] * (EMBED_DIM - 1)}]}
            )
        )
    )

    [vector] = await client.embed(["anything"])

    assert math.isclose(vector[0], 1.0, rel_tol=1e-6)


async def test_a_zero_vector_does_not_divide_by_zero() -> None:
    client = EmbedClient("http://127.0.0.1:8770")
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"data": [{"index": 0, "embedding": [0.0] * EMBED_DIM}]}
            )
        )
    )

    [vector] = await client.embed(["anything"])

    assert set(vector) == {0.0}


async def test_normalisation_can_be_turned_off() -> None:
    client = EmbedClient("http://127.0.0.1:8770", normalize=False)
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"data": [{"index": 0, "embedding": [3.0] + [0.0] * (EMBED_DIM - 1)}]}
            )
        )
    )

    [vector] = await client.embed(["anything"])

    assert vector[0] == 3.0


async def test_a_server_error_is_raised_rather_than_returning_garbage() -> None:
    client = EmbedClient("http://127.0.0.1:8770")
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(503, text="overloaded"))
    )

    with pytest.raises(httpx.HTTPStatusError):
        await client.embed(["anything"])


async def test_closing_the_client_is_safe_to_call() -> None:
    client = EmbedClient("http://127.0.0.1:8770")

    await client.aclose()

    assert client._client.is_closed
