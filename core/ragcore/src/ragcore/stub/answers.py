"""Answer generation.

Scripted by default so every UI state is reproducible on demand. Set
RAGCORE_LLM to an OpenAI-compatible base URL and the same route streams from a
real model instead, with retrieval still served from fixtures.

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

from ragcore.api.schemas import RetrievedChunk
from ragcore.config import Config

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


async def llm_stream(
    config: Config, question: str, chunks: list[RetrievedChunk]
) -> AsyncIterator[str]:
    """Stream from a real OpenAI-compatible endpoint."""
    context = "\n\n".join(
        f"[{c.doc_id}:{c.page_start}] {c.doc_title} — {c.section_path or ''}\n{c.text}"
        for c in chunks
    )
    payload = {
        "model": config.llm_model or "local-model",
        "stream": True,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Passages:\n\n{context}\n\nQuestion: {question}"},
        ],
    }
    headers = {"Content-Type": "application/json"}
    if config.llm_api_key:
        headers["Authorization"] = f"Bearer {config.llm_api_key}"

    base = (config.llm_base_url or "").rstrip("/")
    async with (
        httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client,
        client.stream(
            "POST", f"{base}/chat/completions", json=payload, headers=headers
        ) as response,
    ):
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            try:
                delta = json.loads(data)["choices"][0]["delta"]
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            # Reasoning models emit a separate channel; it never reaches the UI.
            piece = delta.get("content")
            if piece:
                yield piece
