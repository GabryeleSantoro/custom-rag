"""Command line entry point.

The Rust shell spawns this with an explicit port, session token and the
hardware budget it measured. Running it by hand works too: every flag has a
sensible default and auth is off when no token is given.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from ragcore import __version__
from ragcore.api.app import create_app
from ragcore.config import Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragcore", description="RAG core sidecar")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="Run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765, help="0 picks a free port")
    serve.add_argument("--token", default="", help="Session token; empty disables auth")
    serve.add_argument("--data-dir", type=Path, default=Path.home() / ".custom-rag")
    serve.add_argument("--ram-mb", type=int, default=None)
    serve.add_argument("--vram-mb", type=int, default=0)
    serve.add_argument("--gpu-backend", default="cpu", choices=["metal", "cuda", "vulkan", "cpu"])
    serve.add_argument("--backend", default="stub", choices=["stub", "real"])
    serve.add_argument("--prod", action="store_true", help="Disable dev-only routes")
    serve.add_argument("--log-level", default="info")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "serve":
        build_parser().print_help()
        return 1

    env = Config.from_env()
    config = Config(
        host=args.host,
        port=args.port,
        token=args.token,
        data_dir=args.data_dir,
        dev_mode=not args.prod,
        vram_mb=args.vram_mb,
        gpu_backend=args.gpu_backend,
        backend=args.backend,
        llm_base_url=env.llm_base_url,
        llm_model=env.llm_model,
        llm_api_key=env.llm_api_key,
    )
    if args.ram_mb:
        config.ram_mb = args.ram_mb

    config.data_dir.mkdir(parents=True, exist_ok=True)

    uvicorn.run(
        create_app(config),
        host=config.host,
        port=config.port,
        log_level=args.log_level,
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
