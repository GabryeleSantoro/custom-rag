"""llm_stream() must talk to whatever connection the user activated: its URL,
its model, its key, its provider routing — and never the environment."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from ragcore.api.schemas import Connection
from ragcore.stub import answers


class _FakeStreamResponse:
    status_code = 200

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStreamContext:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def __aenter__(self) -> _FakeStreamResponse:
        return _FakeStreamResponse(self._lines)

    async def __aexit__(self, *exc) -> None:
        return None


class _FakeClient:
    def __init__(self, captured: dict, lines: list[str]) -> None:
        self._captured = captured
        self._lines = lines

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    def stream(self, method: str, url: str, *, json: dict, headers: dict):
        self._captured["url"] = url
        self._captured["payload"] = json
        self._captured["headers"] = headers
        return _FakeStreamContext(self._lines)


OPENAI_LINES = ['data: {"choices": [{"delta": {"content": "hello"}}]}', "data: [DONE]"]
ANTHROPIC_LINES = [
    'data: {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "hm"}}',
    'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hello"}}',
    'data: {"type": "message_stop"}',
]


def connection(**overrides) -> Connection:
    fields = {
        "id": "conn_1",
        "name": "Test",
        "kind": "openai-compatible",
        "base_url": "http://x/v1",
        "model_id": "gemma-3-27b",
        "max_output_tokens": 4096,
        "thinking": "off",
        "is_remote": True,
        "has_api_key": True,
        "active": True,
        "created_at": datetime.now(tz=UTC),
    }
    return Connection(**{**fields, **overrides})


def run(conn: Connection, api_key: str | None = "sk-test", lines=OPENAI_LINES, **kwargs):
    captured: dict = {}

    async def collect() -> list[str]:
        return [
            piece
            async for piece in answers.llm_stream(
                conn, api_key, "What is reranking?", [], **kwargs
            )
        ]

    original = answers.httpx.AsyncClient
    answers.httpx.AsyncClient = lambda **_: _FakeClient(captured, lines)
    try:
        return asyncio.run(collect()), captured
    finally:
        answers.httpx.AsyncClient = original


def test_openai_compatible_uses_the_connection_url_model_and_key() -> None:
    pieces, captured = run(connection())

    assert pieces == ["hello"]
    assert captured["url"] == "http://x/v1/chat/completions"
    assert captured["payload"]["model"] == "gemma-3-27b"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"


def test_max_tokens_defaults_to_the_connections_output_budget() -> None:
    _, captured = run(connection(max_output_tokens=32000))
    assert captured["payload"]["max_tokens"] == 32000

    _, captured = run(connection(max_output_tokens=32000), max_tokens=512)
    assert captured["payload"]["max_tokens"] == 512


def test_the_given_system_prompt_wins_over_the_chat_default() -> None:
    _, captured = run(connection(), system_prompt="CUSTOM PROMPT")
    assert captured["payload"]["messages"][0] == {"role": "system", "content": "CUSTOM PROMPT"}

    _, captured = run(connection())
    assert captured["payload"]["messages"][0] == {
        "role": "system",
        "content": answers.SYSTEM_PROMPT,
    }


def test_openrouter_provider_routing_is_sent_only_when_the_user_set_it() -> None:
    _, captured = run(connection(provider_sort="price", provider_order=["together"]))
    assert captured["payload"]["provider"] == {"sort": "price", "order": ["together"]}

    _, captured = run(connection())
    assert "provider" not in captured["payload"]


def test_anthropic_uses_its_native_messages_endpoint_and_headers() -> None:
    pieces, captured = run(
        connection(kind="anthropic", base_url=None), lines=ANTHROPIC_LINES
    )

    assert pieces == ["hello"]  # the thinking_delta never reaches the UI
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "sk-test"
    assert "Authorization" not in captured["headers"]
    assert captured["payload"]["system"] == answers.SYSTEM_PROMPT


def test_a_connection_without_a_base_url_fails_loudly_instead_of_scripting() -> None:
    with pytest.raises(RuntimeError):
        run(connection(base_url=None))
