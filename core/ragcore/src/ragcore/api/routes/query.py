"""The query route: retrieve, stream an answer, validate its citations.

Frame order is fixed and the UI depends on it:
    start -> mode -> sources -> token* -> citations -> done
with `error` replacing everything from the point it occurs.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from fastapi import APIRouter, Request

from ragcore.api.deps import ConfigDep, StoreDep
from ragcore.api.schemas import (
    ChatMessage,
    CitationsEvent,
    DoneEvent,
    ErrorEvent,
    ModeEvent,
    Ok,
    QueryRequest,
    SourcesEvent,
    StartEvent,
    TokenEvent,
)
from ragcore.api.sse import frame, sse_response
from ragcore.stub.answers import (
    extract_citations,
    llm_stream,
    parse_directives,
    scripted_stream,
)

router = APIRouter(tags=["query"])

# Questions about the corpus as a whole cannot be answered from six passages.
GLOBAL_HINTS = (
    "across", "overall", "in general", "main themes", "summarize all", "summarise all",
    "every document", "all documents", "what topics", "compare the",
)

_cancelled: set[str] = set()


def route_mode(question: str) -> tuple[str, str]:
    lowered = question.lower()
    hit = next((h for h in GLOBAL_HINTS if h in lowered), None)
    if hit:
        return "global", f'Phrase "{hit}" asks about the corpus, not a passage'
    return "local", "Answerable from individual passages"


@router.post("/query")
async def query(payload: QueryRequest, request: Request, store: StoreDep, config: ConfigDep):
    query_id = payload.query_id or f"q_{uuid.uuid4().hex[:10]}"
    question, directives = parse_directives(payload.q)

    session_id = payload.session_id
    if session_id is None or session_id not in store.sessions:
        session_id = store.create_session(question[:48]).id

    async def events() -> AsyncIterator[str]:
        started = time.perf_counter()
        yield frame("start", StartEvent(query_id=query_id, session_id=session_id))

        mode, reason = route_mode(question)
        if payload.mode != "auto":
            mode, reason = payload.mode, "Set manually"
        yield frame("mode", ModeEvent(mode=mode, reason=reason))

        chunks, latency, candidates = store.retriever.search(
            question,
            settings=store.settings.retrieval,
            filters=payload.filters,
            doc_meta=store.doc_meta(),
        )
        yield frame(
            "sources",
            SourcesEvent(chunks=chunks, candidates=candidates, kept=len(chunks)),
        )

        store.append_message(
            ChatMessage(
                id=store.new_id("msg"),
                session_id=session_id,
                role="user",
                text=question,
                created_at=datetime.now(tz=UTC),
            )
        )

        connection = store.connections.get(store.settings.active_connection_id or "")
        use_llm = bool(config.llm_base_url)
        stream = (
            llm_stream(config, question, chunks)
            if use_llm
            else scripted_stream(question, chunks, directives)
        )

        answer: list[str] = []
        first_token_at: float | None = None
        try:
            async for piece in stream:
                if query_id in _cancelled:
                    break
                if await request.is_disconnected():
                    break
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                answer.append(piece)
                yield frame("token", TokenEvent(text=piece))
        except Exception as exc:  # noqa: BLE001 - reported to the UI as an error frame
            yield frame("error", ErrorEvent(message=str(exc), retryable=True))
            return
        finally:
            _cancelled.discard(query_id)

        text = "".join(answer)
        citations, dropped, grounding = extract_citations(text, chunks)
        yield frame(
            "citations",
            CitationsEvent(citations=citations, dropped=dropped, grounding=grounding),
        )

        latency.llm_first_token_ms = (
            (first_token_at - started) * 1000 if first_token_at else 0.0
        )
        latency.total_ms = (time.perf_counter() - started) * 1000

        message = ChatMessage(
            id=store.new_id("msg"),
            session_id=session_id,
            role="assistant",
            text=text,
            mode=mode,
            citations=citations,
            chunks=chunks,
            grounding=grounding,
            created_at=datetime.now(tz=UTC),
        )
        store.append_message(message)

        yield frame(
            "done",
            DoneEvent(
                message_id=message.id,
                latency=latency,
                connection_id=connection.id if connection else None,
                remote=bool(connection and connection.is_remote),
            ),
        )

    return sse_response(events())


@router.post("/query/{query_id}/cancel", response_model=Ok)
async def cancel_query(query_id: str) -> Ok:
    _cancelled.add(query_id)
    # The generator clears it on its way out; this is only a safety net for a
    # cancel that arrives after the stream already finished.
    async def expire() -> None:
        await asyncio.sleep(30)
        _cancelled.discard(query_id)

    with contextlib.suppress(RuntimeError):
        asyncio.create_task(expire())
    return Ok()
