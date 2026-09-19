"""Hugging Face Hub browsing.

Anonymous public API — no token needed for search, model metadata or the file
tree. (`?blobs=true` does need auth, which is why file sizes come from
`/tree/main` instead.) Fit verdicts are computed here because the Rust shell
already measured the hardware and passes the budget in at spawn time.
"""

from __future__ import annotations

import re
from datetime import datetime

import httpx

from ragcore.api.schemas import FitVerdict, HubFile, HubModel, HubModelDetail, ModelRole
from ragcore.config import Config

HUB = "https://huggingface.co/api"

PIPELINE_BY_ROLE: dict[ModelRole, str] = {
    "embedding": "sentence-similarity",
    "reranking": "text-ranking",
}

# Known-good starting points per role. Search results that match one are
# flagged, and these are what the "recommended for your hardware" list draws on.
CURATED: dict[ModelRole, list[tuple[str, str]]] = {
    "embedding": [
        ("Qwen/Qwen3-Embedding-0.6B-GGUF", "Ships with the app; changing it forces a re-index"),
        ("Qwen/Qwen3-Embedding-4B-GGUF", "Higher recall, needs roughly 4 GB"),
        ("nomic-ai/nomic-embed-text-v1.5-GGUF", "Small and fast on CPU"),
    ],
    "reranking": [
        ("Qwen/Qwen3-Reranker-0.6B-GGUF", "Ships with the app"),
        ("Qwen/Qwen3-Reranker-4B-GGUF", "Best quality; worth it above 8 GB of VRAM"),
        ("BAAI/bge-reranker-v2-m3", "Multilingual alternative"),
    ],
}

_QUANT = re.compile(r"(IQ\d[A-Z_]*|Q\d_[A-Z0-9_]+|Q\d|F16|BF16|F32)", re.IGNORECASE)
# Working memory beyond the weights themselves: KV cache, context, activations.
OVERHEAD = 1.2


def parse_quant(filename: str) -> str | None:
    match = _QUANT.search(filename)
    return match.group(1).upper() if match else None


def verdict(size_bytes: int, config: Config) -> tuple[FitVerdict, str]:
    need_mb = int(size_bytes / (1024 * 1024) * OVERHEAD)
    if config.vram_mb and need_mb <= config.vram_mb:
        return "vram", f"~{need_mb} MB needed, {config.vram_mb} MB of VRAM available"
    usable_ram = int(config.ram_mb * 0.6)
    if need_mb <= usable_ram:
        note = f"~{need_mb} MB needed, fits in RAM but runs on CPU"
        return "ram", note
    return "too-large", f"~{need_mb} MB needed, only {usable_ram} MB usable"


def _to_model(raw: dict, role: ModelRole | None) -> HubModel:
    repo_id = raw.get("id", "")
    author, _, name = repo_id.partition("/")
    tags = raw.get("tags", []) or []
    curated = dict(CURATED.get(role, [])) if role else {}
    updated = raw.get("lastModified") or raw.get("createdAt")

    return HubModel(
        id=repo_id,
        author=author or None,
        name=name or repo_id,
        downloads=raw.get("downloads", 0) or 0,
        likes=raw.get("likes", 0) or 0,
        trending_score=raw.get("trendingScore", 0) or 0,
        pipeline_tag=raw.get("pipeline_tag"),
        license=next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), None),
        tags=[t for t in tags if ":" not in t][:8],
        updated_at=datetime.fromisoformat(updated.replace("Z", "+00:00")) if updated else None,
        gguf="gguf" in tags,
        recommended=repo_id in curated,
        recommended_reason=curated.get(repo_id),
    )


class HubClient:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=8.0))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def search(
        self, *, role: ModelRole | None, query: str | None, sort: str, limit: int
    ) -> list[HubModel]:
        params: dict[str, str | int] = {
            "limit": limit,
            "sort": sort,
            "direction": -1,
            "filter": "gguf",
        }
        if query:
            params["search"] = query
        if role:
            params["pipeline_tag"] = PIPELINE_BY_ROLE[role]

        response = await self._client.get(f"{HUB}/models", params=params)
        response.raise_for_status()
        return [_to_model(raw, role) for raw in response.json()]

    async def recommended(self, role: ModelRole) -> list[HubModel]:
        """Curated picks, kept only when their smallest quant actually fits."""
        models: list[HubModel] = []
        for repo_id, _ in CURATED.get(role, []):
            try:
                detail = await self.detail(repo_id, role)
            except httpx.HTTPError:
                continue
            if any(f.fit in ("vram", "ram") for f in detail.files) or not detail.files:
                models.append(detail.model)
        return models

    async def detail(self, repo_id: str, role: ModelRole | None = None) -> HubModelDetail:
        meta = await self._client.get(f"{HUB}/models/{repo_id}")
        meta.raise_for_status()
        model = _to_model(meta.json(), role)

        tree = await self._client.get(f"{HUB}/models/{repo_id}/tree/main")
        tree.raise_for_status()

        files: list[HubFile] = []
        for entry in tree.json():
            if entry.get("type") != "file":
                continue
            path = entry.get("path", "")
            if not path.lower().endswith(".gguf"):
                continue
            size = entry.get("size", 0) or 0
            fit, note = verdict(size, self.config)
            files.append(
                HubFile(path=path, size_bytes=size, quant=parse_quant(path), fit=fit, fit_note=note)
            )

        files.sort(key=lambda f: f.size_bytes)
        return HubModelDetail(model=model, files=files)

    @staticmethod
    def download_url(repo_id: str, filename: str) -> str:
        return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
