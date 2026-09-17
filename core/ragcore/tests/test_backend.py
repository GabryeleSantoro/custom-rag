"""The seam itself: the factory honours the flag and the app never names a backend."""

from __future__ import annotations

from pathlib import Path

import pytest
from ragcore.backend import build_backend
from ragcore.config import Config


def config_for(tmp_path: Path, backend: str) -> Config:
    return Config(host="127.0.0.1", port=0, token="", data_dir=tmp_path, backend=backend)


def test_stub_is_the_default(tmp_path: Path) -> None:
    assert Config(data_dir=tmp_path).backend == "stub"


def test_factory_builds_the_stub_backend(tmp_path: Path) -> None:
    built = build_backend(config_for(tmp_path, "stub"))

    assert type(built.store).__module__.startswith("ragcore.stub")
    assert built.answerer is not None


def test_unknown_backend_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        build_backend(config_for(tmp_path, "nonsense"))
