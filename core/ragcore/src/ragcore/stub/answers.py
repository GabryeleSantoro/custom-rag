"""Answer generation.

Scripted only while no connection is active, so a fresh install still has every
UI state reproducible on demand. As soon as the user activates a connection in
the app, the same routes stream from that connection's model — nothing here
reads the environment.

Dev affordances, recognised as a prefix on the question:
  !nocite   answer with no citations at all      -> grounding "none"
  !badcite  answer citing a document not retrieved -> one dropped citation
  !error    fail mid-stream                      -> error event
  !slow     stream slowly, for testing cancellation
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections.abc import AsyncIterator

import httpx

from ragcore.api.schemas import Connection, RetrievedChunk

SYSTEM_PROMPT = """You answer questions using only the passages provided.
Cite every claim with a marker of the form [document_id:page] taken from the
passage headers. If the passages do not contain the answer, say so plainly and
cite nothing. Do not invent document ids."""

NOT_FOUND = (
    "Nothing in the indexed documents answers that. The closest passages were about "
    "other topics, so rather than guess, here is what is missing: no document in this "
    "library covers it."
)


def _first_sentence(text: str, limit: int = 240) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    sentence = parts[0] if parts else text
    return sentence if len(sentence) <= limit else sentence[: limit - 1].rstrip() + "…"


def compose_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    """Build a grounded answer out of the retrieved passages themselves."""
    if not chunks:
        return NOT_FOUND

    lead = chunks[0]
    body = [f"{_first_sentence(lead.text)} [{lead.doc_id}:{lead.page_start}]"]

    supporting = chunks[1:4]
    if supporting:
        body.append("")
        for chunk in supporting:
            body.append(f"- {_first_sentence(chunk.text)} [{chunk.doc_id}:{chunk.page_start}]")

    titles = sorted({c.doc_title for c in chunks})
    if len(titles) > 1:
        body.append("")
        body.append(f"Drawn from {len(titles)} documents: {', '.join(titles)}.")
    return "\n".join(body)


async def scripted_stream(
    question: str, chunks: list[RetrievedChunk], directives: set[str]
) -> AsyncIterator[str]:
    """Emit the composed answer word by word, at a plausible pace."""
    if "nocite" in directives:
        text = (
            "The indexed passages point in this direction, but none of them states it "
            "outright, so this answer is not grounded in a specific passage."
        )
    elif "badcite" in directives:
        text = compose_answer(question, chunks) + " [not-a-real-doc:9]"
    else:
        text = compose_answer(question, chunks)

    delay = 0.09 if "slow" in directives else 0.018
    words = text.split(" ")
    for index, word in enumerate(words):
        if "error" in directives and index == min(12, len(words) - 1):
            raise RuntimeError("Generation failed: connection reset by the model server")
        await asyncio.sleep(delay * random.uniform(0.6, 1.5))
        yield word if index == 0 else " " + word


ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"


def _user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(
        f"[{c.doc_id}:{c.page_start}] {c.doc_title} — {c.section_path or ''}\n{c.text}"
        for c in chunks
    )
    return f"Passages:\n\n{context}\n\nQuestion: {question}"


def _request(
    connection: Connection,
    api_key: str | None,
    question: str,
    chunks: list[RetrievedChunk],
    system_prompt: str,
    max_tokens: int,
) -> tuple[str, dict, dict]:
    """The URL, JSON body and headers for one connection's streaming call."""
    user = _user_prompt(question, chunks)
    if connection.kind == "anthropic":
        base = (connection.base_url or ANTHROPIC_BASE_URL).rstrip("/")
        headers = {"Content-Type": "application/json", "anthropic-version": ANTHROPIC_VERSION}
        if api_key:
            headers["x-api-key"] = api_key
        body = {
            "model": connection.model_id,
            "stream": True,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user}],
        }
        return f"{base}/messages", body, headers

    base = (connection.base_url or "").rstrip("/")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {
        "model": connection.model_id,
        "stream": True,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ],
    }
    # OpenRouter reads this; every other OpenAI-compatible server ignores it.
    routing = {
        key: value
        for key, value in (
            ("sort", connection.provider_sort),
            ("order", connection.provider_order),
        )
        if value
    }
    if routing:
        body["provider"] = routing
    return f"{base}/chat/completions", body, headers


def _piece(kind: str, data: str) -> str | None:
    """One text fragment out of one SSE data line, or None if it carries none."""
    try:
        event = json.loads(data)
    except json.JSONDecodeError:
        return None
    if kind == "anthropic":
        # Reasoning blocks arrive as thinking_delta; only text reaches the UI.
        delta = event.get("delta") or {}
        return delta.get("text") if delta.get("type") == "text_delta" else None
    try:
        # Reasoning models emit a separate channel; it never reaches the UI.
        return (event["choices"][0]["delta"] or {}).get("content")
    except (KeyError, IndexError):
        return None


async def llm_stream(
    connection: Connection,
    api_key: str | None,
    question: str,
    chunks: list[RetrievedChunk],
    *,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
) -> AsyncIterator[str]:
    """Stream from the connection the user activated, and from nothing else."""
    if connection.kind != "anthropic" and not connection.base_url:
        raise RuntimeError(f"Connection {connection.name!r} has no base URL")

    url, body, headers = _request(
        connection,
        api_key,
        question,
        chunks,
        system_prompt or SYSTEM_PROMPT,
        max_tokens or connection.max_output_tokens,
    )
    async with (
        httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client,
        client.stream("POST", url, json=body, headers=headers) as response,
    ):
        if response.status_code >= 400:
            detail = (await response.aread()).decode(errors="replace").strip()[:500]
            raise RuntimeError(
                f"{connection.name} returned HTTP {response.status_code}: {detail}"
            )
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            piece = _piece(connection.kind, data)
            if piece:
                yield piece
