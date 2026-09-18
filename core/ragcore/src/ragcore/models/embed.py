"""Embedding over llama-server's OpenAI-compatible endpoint.

The server is started with `--embedding --pooling last`. Normalisation is done
here regardless of what the server returns, because the index stores unit
vectors and cosine similarity on unit vectors is a dot product.
"""

from __future__ import annotations

import httpx
import numpy as np

EMBED_DIM = 1024


class EmbedClient:
    dim = EMBED_DIM

    def __init__(
        self,
        base_url: str,
        *,
        model: str = "qwen3-embedding",
        timeout: float = 120.0,
        normalize: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.normalize = normalize
        self._client = httpx.AsyncClient(timeout=timeout)

    async def _post(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.post(
            f"{self.base_url}/v1/embeddings",
            json={"input": texts, "model": self.model},
        )
        response.raise_for_status()
        rows = sorted(response.json()["data"], key=lambda row: row["index"])
        return [row["embedding"] for row in rows]

    async def embed(self, texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            out.extend(await self._post(texts[start : start + batch_size]))
        if not self.normalize:
            return out
        matrix = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (matrix / norms).tolist()

    async def aclose(self) -> None:
        await self._client.aclose()
