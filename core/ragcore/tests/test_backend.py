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


def test_the_ram_probe_falls_back_when_the_platform_will_not_say(monkeypatch) -> None:
    """Windows has no sysconf; a missing figure must not stop the sidecar booting."""
    from ragcore import config as config_module

    monkeypatch.setattr(
        config_module.os, "sysconf", lambda _: (_ for _ in ()).throw(AttributeError())
    )

    assert config_module._default_ram_mb() == 8192


def test_the_ram_probe_reports_a_plausible_figure() -> None:
    from ragcore.config import _default_ram_mb

    assert _default_ram_mb() >= 512


def test_the_backend_hands_the_api_layer_one_of_everything(tmp_path: Path) -> None:
    built = build_backend(config_for(tmp_path, "stub"))

    assert built.store is not None and built.jobs is not None
    assert built.hub is not None and built.answerer is not None


def test_the_store_the_answerer_reads_is_the_one_the_api_layer_gets(tmp_path: Path) -> None:
    """A second store would answer from a different index than /documents lists."""
    built = build_backend(config_for(tmp_path, "stub"))

    assert built.answerer.store is built.store
