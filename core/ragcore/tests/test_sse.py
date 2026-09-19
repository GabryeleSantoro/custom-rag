"""The wire format the Rust proxy parses. Boring on purpose."""

from __future__ import annotations

import json

from ragcore.api.schemas import TokenEvent
from ragcore.api.sse import SSE_HEADERS, frame


def test_a_frame_is_an_event_line_a_data_line_and_a_blank_line() -> None:
    raw = frame("token", {"text": "hello"})

    assert raw == 'event: token\ndata: {"text": "hello"}\n\n'


def test_a_pydantic_payload_is_serialised_by_the_model() -> None:
    raw = frame("token", TokenEvent(text="hi"))

    assert json.loads(raw.split("data: ", 1)[1]) == {"text": "hi"}


def test_a_payload_never_spans_more_than_one_line() -> None:
    """A newline inside data would split one frame into two on the Rust side."""
    raw = frame("token", {"text": "first\nsecond"})

    assert raw.count("\n") == 3
    assert json.loads(raw.split("data: ", 1)[1])["text"] == "first\nsecond"


def test_a_value_json_cannot_encode_falls_back_to_its_string_form() -> None:
    from datetime import UTC, datetime

    raw = frame("done", {"at": datetime(2026, 3, 1, tzinfo=UTC)})

    assert "2026-03-01" in raw


def test_the_headers_defeat_intermediate_buffering() -> None:
    assert SSE_HEADERS["Cache-Control"] == "no-cache, no-transform"
    assert SSE_HEADERS["X-Accel-Buffering"] == "no"
