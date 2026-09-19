"""The seam itself: the factory honours the flag and the app never names a backend."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from ragcore.api.schemas import Connection
from ragcore.backend import build_backend
from ragcore.config import Config
from ragcore.ports import AnswerEngine


def config_for(tmp_path: Path, backend: str) -> Config:
    return Config(host="127.0.0.1", port=0, token="", data_dir=tmp_path, backend=backend)


def test_stub_is_the_default(tmp_path: Path) -> None:
    assert Config(data_dir=tmp_path).backend == "stub"


def test_factory_builds_the_stub_backend(tmp_path: Path) -> None:
    built = build_backend(config_for(tmp_path, "stub"))

    assert type(built.store).__module__.startswith("ragcore.stub")
    assert isinstance(built.answerer, AnswerEngine)


def test_unknown_backend_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        build_backend(config_for(tmp_path, "nonsense"))


def test_the_active_connection_decides_who_answers(tmp_path: Path, monkeypatch) -> None:
    """The scripted stub answers only while no connection is active — never
    because an environment variable is missing."""
    monkeypatch.setenv("RAGCORE_LLM", "https://openrouter.ai/api/v1")
    built = build_backend(config_for(tmp_path, "stub"))

    assert _generator_name(built.answerer) == "scripted_stream"

    connection = built.store.connections.setdefault(
        "conn_1",
        Connection(
            id="conn_1", name="OpenRouter", kind="openai-compatible",
            base_url="https://openrouter.ai/api/v1", model_id="google/gemma-3-27b-it",
            max_output_tokens=4096, thinking="off",
            is_remote=True, has_api_key=True, active=True, created_at=datetime.now(tz=UTC),
        ),
    )
    built.store.settings.active_connection_id = connection.id

    assert _generator_name(built.answerer) == "llm_stream"


def _generator_name(answerer) -> str:
    """Which coroutine the engine handed back. Never iterated, so nothing runs."""
    return answerer.stream("q", [], set()).__qualname__
