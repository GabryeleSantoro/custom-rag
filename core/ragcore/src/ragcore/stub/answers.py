"""The scripted answer, used while no connection is active.

A fresh install still has every UI state reproducible on demand. As soon as the
user activates a connection, ``ragcore.llm`` answers instead.

Dev affordances, recognised as a prefix on the question:
  !nocite   answer with no citations at all      -> grounding "none"
  !badcite  answer citing a document not retrieved -> one dropped citation
  !error    fail mid-stream                      -> error event
  !slow     stream slowly, for testing cancellation
"""

from __future__ import annotations

import asyncio
import random
import re
from collections.abc import AsyncIterator

from ragcore.api.schemas import RetrievedChunk
from ragcore.llm import LlmError

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
            raise LlmError(
                "generation_failed", "Generation failed: connection reset by the model server"
            )
        await asyncio.sleep(delay * random.uniform(0.6, 1.5))
        yield word if index == 0 else " " + word

