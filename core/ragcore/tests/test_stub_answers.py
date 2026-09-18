"""llm_stream() must use whatever system prompt the caller gives it, and fall
back to the chat SYSTEM_PROMPT when the caller gives none."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
from ragcore.config import Config
from ragcore.stub import answers


class _FakeStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStreamContext:
    async def __aenter__(self) -> _FakeStreamResponse:
        return _FakeStreamResponse(
            ['data: {"choices": [{"delta": {"content": "hello"}}]}', "data: [DONE]"]
        )

    async def __aexit__(self, *exc) -> None:
        return None


class _FakeClient:
    def __init__(self, captured: dict) -> None:
        self._captured = captured

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    def stream(self, method: str, url: str, *, json: dict, headers: dict):
        self._captured["method"] = method
        self._captured["url"] = url
        self._captured["payload"] = json
        return _FakeStreamContext()


def _run(config: Config, system_prompt: str | None) -> tuple[list[str], dict]:
    import asyncio

    captured: dict = {}

    async def collect() -> list[str]:
        return [
            piece
            async for piece in answers.llm_stream(
                config, "What is reranking?", [], system_prompt=system_prompt
            )
        ]

    original_client = answers.httpx.AsyncClient
    answers.httpx.AsyncClient = lambda **_: _FakeClient(captured)
    try:
        pieces = asyncio.run(collect())
    finally:
        answers.httpx.AsyncClient = original_client
    return pieces, captured


def test_llm_stream_uses_the_given_system_prompt(tmp_path: Path) -> None:
    config = Config(
        host="127.0.0.1", port=0, token="", data_dir=tmp_path, llm_base_url="http://x/v1"
    )

    pieces, captured = _run(config, "CUSTOM PROMPT")

    assert pieces == ["hello"]
    assert captured["payload"]["messages"][0] == {"role": "system", "content": "CUSTOM PROMPT"}


def test_llm_stream_falls_back_to_the_chat_system_prompt_when_none_given(tmp_path: Path) -> None:
    config = Config(
        host="127.0.0.1", port=0, token="", data_dir=tmp_path, llm_base_url="http://x/v1"
    )

    _, captured = _run(config, None)

    assert captured["payload"]["messages"][0] == {"role": "system", "content": answers.SYSTEM_PROMPT}
