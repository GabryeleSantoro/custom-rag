"""Hugging Face browsing: fit verdicts, quant parsing, and what each call asks for.

No network. The client's transport is swapped for one that answers from a table,
which also lets the tests assert the query parameters the Hub actually receives.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from ragcore.config import Config
from ragcore.stub.hub import CURATED, HubClient, parse_quant, verdict


def config_for(tmp_path: Path, *, ram_mb: int = 16384, vram_mb: int = 0) -> Config:
    return Config(data_dir=tmp_path, ram_mb=ram_mb, vram_mb=vram_mb)


def hub_with(config: Config, handler) -> HubClient:
    """A HubClient whose transport answers from `handler` instead of the network."""
    client = HubClient(config)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


# --------------------------------------------------------------------- quants


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("Qwen3-Embedding-0.6B-Q8_0.gguf", "Q8_0"),
        ("model-q4_k_m.gguf", "Q4_K_M"),
        ("model-IQ3_XXS.gguf", "IQ3_XXS"),
        ("model-f16.gguf", "F16"),
        ("model-BF16.gguf", "BF16"),
        ("model.gguf", None),
    ],
)
def test_quant_is_read_off_the_filename(filename: str, expected: str | None) -> None:
    assert parse_quant(filename) == expected


# ------------------------------------------------------------------- fit badge

GB = 1024 * 1024 * 1024


def test_a_model_that_fits_in_vram_says_so(tmp_path: Path) -> None:
    fit, note = verdict(2 * GB, config_for(tmp_path, vram_mb=8192))

    assert fit == "vram"
    assert "VRAM" in note


def test_a_model_too_big_for_vram_falls_back_to_ram(tmp_path: Path) -> None:
    fit, _ = verdict(6 * GB, config_for(tmp_path, ram_mb=32768, vram_mb=4096))

    assert fit == "ram"


def test_a_model_larger_than_the_usable_ram_is_too_large(tmp_path: Path) -> None:
    # Only 60% of RAM counts as usable, so 8 GB of weights does not fit in 8 GB.
    fit, note = verdict(8 * GB, config_for(tmp_path, ram_mb=8192))

    assert fit == "too-large"
    assert "usable" in note


def test_the_overhead_factor_is_charged_against_the_budget(tmp_path: Path) -> None:
    """1 GB of weights needs more than 1 GB: KV cache and activations count too."""
    _, note = verdict(1 * GB, config_for(tmp_path, vram_mb=8192))

    assert "~1228 MB needed" in note


# ---------------------------------------------------------------------- search


async def test_search_filters_to_gguf_and_the_roles_pipeline(tmp_path: Path) -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.url.params)
        return httpx.Response(200, json=[])

    await hub_with(config_for(tmp_path), handler).search(
        role="reranking", query="qwen", sort="likes", limit=7
    )

    assert seen["filter"] == "gguf"
    assert seen["pipeline_tag"] == "text-ranking"
    assert seen["search"] == "qwen"
    assert (seen["sort"], seen["limit"], seen["direction"]) == ("likes", "7", "-1")


async def test_search_without_a_role_sends_no_pipeline_filter(tmp_path: Path) -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.url.params)
        return httpx.Response(200, json=[])

    await hub_with(config_for(tmp_path), handler).search(
        role=None, query=None, sort="downloads", limit=24
    )

    assert "pipeline_tag" not in seen
    assert "search" not in seen


async def test_search_maps_hub_fields_onto_the_ui_model(tmp_path: Path) -> None:
    raw = {
        "id": "Qwen/Qwen3-Embedding-0.6B-GGUF",
        "downloads": 1234,
        "likes": 56,
        "trendingScore": 7,
        "pipeline_tag": "sentence-similarity",
        "tags": ["gguf", "license:apache-2.0", "text-embedding", "region:us"],
        "lastModified": "2026-02-01T10:00:00.000Z",
    }

    [model] = await hub_with(
        config_for(tmp_path), lambda _: httpx.Response(200, json=[raw])
    ).search(role="embedding", query=None, sort="downloads", limit=1)

    assert (model.author, model.name) == ("Qwen", "Qwen3-Embedding-0.6B-GGUF")
    assert model.license == "apache-2.0"
    assert model.gguf is True
    # Namespaced tags are Hub plumbing, not something to show the user.
    assert model.tags == ["gguf", "text-embedding"]
    assert model.updated_at is not None and model.updated_at.year == 2026
    assert model.recommended is True
    assert model.recommended_reason == dict(CURATED["embedding"])[model.id]


async def test_a_model_outside_the_curated_list_is_not_flagged(tmp_path: Path) -> None:
    [model] = await hub_with(
        config_for(tmp_path), lambda _: httpx.Response(200, json=[{"id": "someone/random-GGUF"}])
    ).search(role="embedding", query=None, sort="downloads", limit=1)

    assert model.recommended is False
    assert model.recommended_reason is None


async def test_a_model_without_a_timestamp_has_no_updated_at(tmp_path: Path) -> None:
    [model] = await hub_with(
        config_for(tmp_path), lambda _: httpx.Response(200, json=[{"id": "a/b"}])
    ).search(role=None, query=None, sort="downloads", limit=1)

    assert model.updated_at is None


# ---------------------------------------------------------------------- detail


def _detail_handler(files: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/tree/main"):
            return httpx.Response(200, json=files)
        return httpx.Response(200, json={"id": "Qwen/Qwen3-Embedding-0.6B-GGUF", "tags": ["gguf"]})

    return handler


async def test_detail_keeps_only_gguf_files_and_sorts_them_smallest_first(
    tmp_path: Path,
) -> None:
    detail = await hub_with(
        config_for(tmp_path),
        _detail_handler(
            [
                {"type": "file", "path": "README.md", "size": 10},
                {"type": "directory", "path": "subdir"},
                {"type": "file", "path": "model-Q8_0.gguf", "size": 3 * GB},
                {"type": "file", "path": "model-Q4_K_M.gguf", "size": 1 * GB},
            ]
        ),
    ).detail("Qwen/Qwen3-Embedding-0.6B-GGUF", "embedding")

    assert [f.path for f in detail.files] == ["model-Q4_K_M.gguf", "model-Q8_0.gguf"]
    assert [f.quant for f in detail.files] == ["Q4_K_M", "Q8_0"]
    assert all(f.fit_note for f in detail.files)


async def test_detail_carries_a_fit_verdict_per_file(tmp_path: Path) -> None:
    detail = await hub_with(
        config_for(tmp_path, ram_mb=8192),
        _detail_handler(
            [
                {"type": "file", "path": "small-Q4_K_M.gguf", "size": 1 * GB},
                {"type": "file", "path": "huge-F16.gguf", "size": 40 * GB},
            ]
        ),
    ).detail("Qwen/Qwen3-Embedding-0.6B-GGUF")

    fits = {f.path: f.fit for f in detail.files}
    assert fits["small-Q4_K_M.gguf"] == "ram"
    assert fits["huge-F16.gguf"] == "too-large"


async def test_a_missing_repo_raises_for_the_route_to_translate(tmp_path: Path) -> None:
    client = hub_with(config_for(tmp_path), lambda _: httpx.Response(404, json={}))

    with pytest.raises(httpx.HTTPStatusError):
        await client.detail("nobody/nothing")


# ----------------------------------------------------------------- recommended


async def test_recommended_drops_a_pick_that_cannot_run_here(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/tree/main"):
            size = 1 * GB if "0.6B" in request.url.path else 40 * GB
            return httpx.Response(200, json=[{"type": "file", "path": "m-Q8_0.gguf", "size": size}])
        return httpx.Response(200, json={"id": request.url.path.removeprefix("/api/models/")})

    models = await hub_with(config_for(tmp_path, ram_mb=8192), handler).recommended("embedding")

    assert [m.id for m in models] == ["Qwen/Qwen3-Embedding-0.6B-GGUF"]


async def test_recommended_skips_a_pick_the_hub_cannot_answer_for(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "Reranker-0.6B" in str(request.url):
            raise httpx.ConnectError("boom", request=request)
        if request.url.path.endswith("/tree/main"):
            return httpx.Response(200, json=[])
        return httpx.Response(200, json={"id": request.url.path.removeprefix("/api/models/")})

    models = await hub_with(config_for(tmp_path), handler).recommended("reranking")

    assert "Qwen/Qwen3-Reranker-0.6B-GGUF" not in [m.id for m in models]
    assert models, "the other curated picks still come through"


async def test_a_repo_with_no_gguf_files_is_still_recommended(tmp_path: Path) -> None:
    """No file list is not evidence that nothing fits; only a too-large one is."""
    models = await hub_with(config_for(tmp_path), _detail_handler([])).recommended("embedding")

    assert len(models) == len(CURATED["embedding"])


def test_the_download_url_points_at_the_resolved_blob() -> None:
    assert HubClient.download_url("Qwen/Qwen3-Embedding-0.6B-GGUF", "m-Q8_0.gguf") == (
        "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/m-Q8_0.gguf"
    )
