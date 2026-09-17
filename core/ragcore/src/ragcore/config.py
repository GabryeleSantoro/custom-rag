"""Runtime configuration, assembled from CLI flags with env fallbacks.

The Rust shell passes everything explicitly at spawn time. The env fallbacks
exist so `uv run ragcore serve` on its own is still usable during development.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_ram_mb() -> int:
    """Total physical RAM, best effort, without pulling in psutil."""
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 * 1024))
    except (ValueError, OSError, AttributeError):
        return 8192


@dataclass(slots=True)
class Config:
    host: str = "127.0.0.1"
    port: int = 8765
    token: str = ""
    data_dir: Path = field(default_factory=lambda: Path.home() / ".custom-rag")
    dev_mode: bool = True
    ram_mb: int = field(default_factory=_default_ram_mb)
    vram_mb: int = 0
    gpu_backend: str = "cpu"
    """An OpenAI-compatible base URL. Set it and /query streams from a real model."""
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: str | None = None

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            host=os.getenv("RAGCORE_HOST", "127.0.0.1"),
            port=int(os.getenv("RAGCORE_PORT", "8765")),
            token=os.getenv("RAGCORE_TOKEN", ""),
            data_dir=Path(os.getenv("RAGCORE_DATA_DIR", str(Path.home() / ".custom-rag"))),
            dev_mode=os.getenv("RAGCORE_DEV", "1") != "0",
            llm_base_url=os.getenv("RAGCORE_LLM") or None,
            llm_model=os.getenv("RAGCORE_LLM_MODEL") or None,
            llm_api_key=os.getenv("RAGCORE_LLM_API_KEY") or None,
        )
