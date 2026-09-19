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


# ------------------------------------------------------- what reaches the UI


def test_only_text_deltas_reach_the_ui() -> None:
    """Anthropic reasoning arrives as thinking_delta and must be dropped."""
    pieces, _ = run(connection(kind="anthropic"), lines=ANTHROPIC_LINES)

    assert pieces == ["hello"]


def test_an_openai_reasoning_channel_is_dropped_too() -> None:
    pieces, _ = run(
        connection(),
        lines=[
            'data: {"choices": [{"delta": {"reasoning": "thinking out loud"}}]}',
            'data: {"choices": [{"delta": {"content": "answer"}}]}',
            "data: [DONE]",
        ],
    )

    assert pieces == ["answer"]


def test_a_line_that_is_not_json_is_skipped_rather_than_crashing() -> None:
    pieces, _ = run(
        connection(),
        lines=["data: not-json", 'data: {"choices": [{"delta": {"content": "ok"}}]}'],
    )

    assert pieces == ["ok"]


def test_lines_outside_the_data_channel_are_ignored() -> None:
    pieces, _ = run(
        connection(),
        lines=[": keep-alive", "", 'data: {"choices": [{"delta": {"content": "ok"}}]}'],
    )

    assert pieces == ["ok"]


def test_a_chunk_with_no_choices_is_skipped() -> None:
    pieces, _ = run(
        connection(),
        lines=['data: {"choices": []}', 'data: {"choices": [{"delta": {"content": "ok"}}]}'],
    )

    assert pieces == ["ok"]


def test_a_null_delta_is_skipped() -> None:
    pieces, _ = run(
        connection(),
        lines=[
            'data: {"choices": [{"delta": null}]}',
            'data: {"choices": [{"delta": {"content": "ok"}}]}',
        ],
    )

    assert pieces == ["ok"]


def test_the_done_sentinel_ends_the_stream() -> None:
    pieces, _ = run(
        connection(),
        lines=[
            'data: {"choices": [{"delta": {"content": "first"}}]}',
            "data: [DONE]",
            'data: {"choices": [{"delta": {"content": "never"}}]}',
        ],
    )

    assert pieces == ["first"]


def test_an_http_error_names_the_connection_and_the_status() -> None:
    class _ErrorResponse:
        status_code = 429

        async def aread(self) -> bytes:
            return b'{"error": "rate limited"}'

        async def aiter_lines(self):  # pragma: no cover - never reached
            yield ""

    class _ErrorContext:
        async def __aenter__(self):
            return _ErrorResponse()

        async def __aexit__(self, *exc) -> None:
            return None

    class _ErrorClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc) -> None:
            return None

        def stream(self, *args, **kwargs):
            return _ErrorContext()

    async def collect() -> list[str]:
        return [
            piece
            async for piece in answers.llm_stream(
                connection(name="OpenRouter"), "sk", "q", []
            )
        ]

    original = answers.httpx.AsyncClient
    answers.httpx.AsyncClient = lambda **_: _ErrorClient()
    try:
        with pytest.raises(RuntimeError, match="OpenRouter returned HTTP 429"):
            asyncio.run(collect())
    finally:
        answers.httpx.AsyncClient = original


def test_a_connection_with_no_key_sends_no_auth_header() -> None:
    _, captured = run(connection(), api_key=None)

    assert "Authorization" not in captured["headers"]


def test_anthropic_without_a_key_sends_no_api_key_header() -> None:
    _, captured = run(connection(kind="anthropic"), api_key=None, lines=ANTHROPIC_LINES)

    assert "x-api-key" not in captured["headers"]
    assert captured["headers"]["anthropic-version"] == "2023-06-01"


def test_the_passages_are_handed_over_as_labelled_context() -> None:
    from ragcore.api.schemas import RetrievedChunk

    chunk = RetrievedChunk(
        chunk_id="c1", doc_id="rerank", doc_title="Reranking", page_start=2, page_end=2,
        section_path="Reranking > Why", text="A cross encoder rescores candidates.",
    )

    captured: dict = {}

    async def collect() -> None:
        async for _ in answers.llm_stream(connection(), "sk", "Why rerank?", [chunk]):
            pass

    original = answers.httpx.AsyncClient
    answers.httpx.AsyncClient = lambda **_: _FakeClient(captured, OPENAI_LINES)
    try:
        asyncio.run(collect())
    finally:
        answers.httpx.AsyncClient = original

    user = captured["payload"]["messages"][-1]["content"]
    assert "[rerank:2]" in user, "the marker the model must cite with"
    assert "A cross encoder rescores candidates." in user
    assert user.endswith("Question: Why rerank?")


# -------------------------------------------------------- the scripted answer


def test_the_scripted_answer_cites_the_passage_it_leads_with() -> None:
    from ragcore.api.schemas import RetrievedChunk

    chunks = [
        RetrievedChunk(
            chunk_id="c1", doc_id="rerank", doc_title="Reranking", page_start=2, page_end=2,
            text="Reranking reorders candidates. It runs after fusion.",
        ),
        RetrievedChunk(
            chunk_id="c2", doc_id="fusion", doc_title="Fusion", page_start=1, page_end=1,
            text="Reciprocal rank fusion merges two rankings.",
        ),
    ]

    text = answers.compose_answer("Why rerank?", chunks)

    assert text.startswith("Reranking reorders candidates. [rerank:2]")
    assert "[fusion:1]" in text
    assert "Drawn from 2 documents" in text


def test_the_scripted_answer_says_so_when_nothing_was_retrieved() -> None:
    assert answers.compose_answer("Why rerank?", []) == answers.NOT_FOUND


def test_a_single_document_answer_skips_the_provenance_line() -> None:
    from ragcore.api.schemas import RetrievedChunk

    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}", doc_id="rerank", doc_title="Reranking", page_start=i, page_end=i,
            text=f"Sentence {i}. More text.",
        )
        for i in (1, 2)
    ]

    assert "Drawn from" not in answers.compose_answer("Why rerank?", chunks)


def test_a_very_long_sentence_is_elided_rather_than_dumped() -> None:
    from ragcore.api.schemas import RetrievedChunk

    chunk = RetrievedChunk(
        chunk_id="c1", doc_id="long", doc_title="Long", page_start=1, page_end=1,
        text="word " * 200 + ".",
    )

    lead = answers.compose_answer("q", [chunk]).split(" [long:1]")[0]

    assert len(lead) <= 240
    assert lead.endswith("…")
