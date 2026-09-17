"""Fixtures for the stub core's contract tests.

Every test runs against a real app instance with auth on, because the token
middleware is part of the contract the Rust shell depends on.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from ragcore.api.app import create_app
from ragcore.config import Config

TOKEN = "test-token"


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    config = Config(
        host="127.0.0.1",
        port=0,
        token=TOKEN,
        data_dir=tmp_path,
        dev_mode=True,
        ram_mb=16384,
        vram_mb=0,
        gpu_backend="cpu",
    )
    with TestClient(create_app(config)) as test_client:
        test_client.headers["Authorization"] = f"Bearer {TOKEN}"
        yield test_client


@pytest.fixture
def read_events() -> Callable[..., list[tuple[str, dict]]]:
    """Collect an SSE body into (event, payload) pairs, in arrival order."""
    return _read_events


def _read_events(response) -> list[tuple[str, dict]]:
    """Collect an SSE body into (event, payload) pairs, in arrival order."""
    events: list[tuple[str, dict]] = []
    name: str | None = None
    for raw in response.iter_lines():
        line = raw.decode() if isinstance(raw, bytes) else raw
        if line.startswith("event:"):
            name = line.removeprefix("event:").strip()
        elif line.startswith("data:") and name is not None:
            events.append((name, json.loads(line.removeprefix("data:").strip())))
            name = None
    return events
