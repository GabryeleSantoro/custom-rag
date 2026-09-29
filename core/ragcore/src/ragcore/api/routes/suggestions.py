"""Starter questions for the chat screen, built from the headings of the selected documents."""

from __future__ import annotations

from itertools import zip_longest

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ragcore.api.deps import StoreDep

router = APIRouter(prefix="/suggestions", tags=["suggestions"])

# ponytail: English template over the doc's own headings; no model involved (a reranker only
# scores passages, it can't write). Swap in a generation call if phrasing matters.
TEMPLATES = (
    "What does the document say about {}?",
    "Can you explain {}?",
    "What are the key points of {}?",
)


class SuggestionRequest(BaseModel):
    doc_ids: list[str] = Field(default_factory=list)


class Suggestions(BaseModel):
    # `topics` is what the UI phrases in its own language; `questions` stays as the
    # English fallback for clients that do not.
    questions: list[str]
    topics: list[str]


@router.post("", response_model=Suggestions)
def suggest(payload: SuggestionRequest, store: StoreDep):
    per_doc: list[list[str]] = []
    for doc_id in payload.doc_ids:
        document = store.documents.get(doc_id)
        content = store.content(doc_id) if document else None
        if document is None or content is None:
            continue
        headings = {
            page.section_path.split(" > ")[-1].strip(): None
            for page in content.pages
            if page.section_path
        }
        per_doc.append([h for h in headings if h and h != document.title] or [document.title])
    # Round-robin so one big document doesn't crowd out the others.
    topics = list(dict.fromkeys(t for row in zip_longest(*per_doc) for t in row if t))[:3]
    return Suggestions(
        questions=[TEMPLATES[i].format(f"“{t}”") for i, t in enumerate(topics)],
        topics=topics,
    )
