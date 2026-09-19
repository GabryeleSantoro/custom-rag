"""Runtime configuration, assembled from CLI flags.

The Rust shell passes everything explicitly at spawn time. Nothing about
generation lives here: which model answers, where it lives and with which key
is decided entirely by the connection the user activates in the app.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


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
    backend: Literal["stub", "real"] = "stub"
