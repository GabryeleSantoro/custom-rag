"""The entry point the Rust shell spawns: flags in, a configured server out."""

from __future__ import annotations

from pathlib import Path

import pytest
from ragcore import __version__
from ragcore.cli import build_parser, main
from ragcore.config import Config


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Capture the assembled Config and the uvicorn kwargs, without binding a socket."""
    calls: list[dict] = []

    def fake_create_app(config: Config) -> str:
        calls.append({"config": config})
        return "app-sentinel"

    def fake_run(app, **kwargs) -> None:
        calls[-1].update(kwargs, app=app)

    monkeypatch.setattr("ragcore.cli.create_app", fake_create_app)
    monkeypatch.setattr("ragcore.cli.uvicorn.run", fake_run)
    return calls


def test_no_subcommand_prints_help_and_fails(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 1
    assert "serve" in capsys.readouterr().out


def test_version_flag_reports_the_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == __version__


def test_serve_passes_host_port_and_log_level_to_uvicorn(
    tmp_path: Path, served: list[dict]
) -> None:
    assert main(["serve", "--host", "0.0.0.0", "--port", "9999", "--data-dir", str(tmp_path),
                 "--log-level", "warning"]) == 0

    [call] = served
    assert call["host"] == "0.0.0.0"
    assert call["port"] == 9999
    assert call["log_level"] == "warning"
    # Access logs would print the session token's caller on every request.
    assert call["access_log"] is False


def test_serve_creates_the_data_directory(tmp_path: Path, served: list[dict]) -> None:
    data_dir = tmp_path / "nested" / "custom-rag"

    main(["serve", "--data-dir", str(data_dir)])

    assert data_dir.is_dir()


def test_the_shells_hardware_budget_reaches_the_app(
    tmp_path: Path, served: list[dict]
) -> None:
    main(["serve", "--data-dir", str(tmp_path), "--ram-mb", "65536", "--vram-mb", "24576",
          "--gpu-backend", "cuda"])

    config = served[0]["config"]
    assert (config.ram_mb, config.vram_mb, config.gpu_backend) == (65536, 24576, "cuda")


def test_ram_mb_falls_back_to_the_probe_when_not_given(
    tmp_path: Path, served: list[dict]
) -> None:
    main(["serve", "--data-dir", str(tmp_path)])

    assert served[0]["config"].ram_mb > 0


def test_prod_turns_dev_mode_off(tmp_path: Path, served: list[dict]) -> None:
    main(["serve", "--data-dir", str(tmp_path), "--prod"])

    assert served[0]["config"].dev_mode is False


def test_dev_mode_is_the_default(tmp_path: Path, served: list[dict]) -> None:
    main(["serve", "--data-dir", str(tmp_path)])

    assert served[0]["config"].dev_mode is True


def test_an_unknown_gpu_backend_is_rejected() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["serve", "--gpu-backend", "opencl"])


def test_an_unknown_backend_is_rejected() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["serve", "--backend", "magic"])


def test_defaults_match_what_the_shell_relies_on() -> None:
    args = build_parser().parse_args(["serve"])

    assert (args.host, args.port, args.token) == ("127.0.0.1", 8765, "")
    assert (args.backend, args.gpu_backend, args.vram_mb) == ("stub", "cpu", 0)
    assert args.prod is False
