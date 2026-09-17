"""Turns the fixture folder into documents, pages and chunks.

Deliberately naive: split on markdown headings for pages and on blank lines for
chunks. It only has to be structurally right, because the real implementation
(headings-then-size, parent/child, tokenizer counts) lands in the ingestion
phase and replaces this wholesale.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

MIME_BY_EXT = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".html": "text/html",
    ".csv": "text/csv",
}

_HEADING = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)
_WORD = re.compile(r"[a-z0-9]+")


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    page: int
    section_path: str
    text: str
    char_start: int
    char_end: int
    tokens: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Page:
    page: int
    section_path: str
    text: str


@dataclass(slots=True)
class LoadedDoc:
    doc_id: str
    path: Path
    title: str
    ext: str
    mime: str
    size_bytes: int
    mtime: datetime
    pages: list[Page]
    chunks: list[Chunk]
    keywords: list[str]


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def find_fixture_dir(start: Path | None = None) -> Path | None:
    """Walk up from this file looking for the repo's fixture corpus."""
    here = (start or Path(__file__)).resolve()
    for parent in here.parents:
        candidate = parent / "fixtures" / "docs"
        if candidate.is_dir():
            return candidate
    return None


def _split_pages(text: str) -> list[Page]:
    """One page per `##` section, with anything before the first one as page 1."""
    matches = [m for m in _HEADING.finditer(text) if len(m.group(1)) == 2]
    if not matches:
        return [Page(page=1, section_path="", text=text.strip())]

    pages: list[Page] = []
    preamble = text[: matches[0].start()].strip()
    title_line = preamble.splitlines()[0] if preamble else ""
    doc_title = title_line.lstrip("# ").strip()

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        heading = match.group(2).strip()
        section = f"{doc_title} > {heading}" if doc_title else heading
        pages.append(Page(page=index + 1, section_path=section, text=f"## {heading}\n\n{body}"))
    return pages


def _split_chunks(doc_id: str, doc_title: str, pages: list[Page]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        offset = 0
        body = page.text
        for index, para in enumerate(p for p in body.split("\n\n") if p.strip()):
            start = body.index(para, offset)
            offset = start + len(para)
            if para.lstrip().startswith("##"):
                continue
            text = " ".join(para.split())
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#p{page.page}c{index}",
                    doc_id=doc_id,
                    doc_title=doc_title,
                    page=page.page,
                    section_path=page.section_path,
                    text=text,
                    char_start=start,
                    char_end=start + len(para),
                    tokens=tokenize(text),
                )
            )
    return chunks


def load_document(path: Path) -> LoadedDoc:
    raw = path.read_text(encoding="utf-8", errors="replace")
    stat = path.stat()
    doc_id = path.stem
    first_line = raw.lstrip().splitlines()[0] if raw.strip() else path.stem
    title = first_line.lstrip("# ").strip() or path.stem.replace("-", " ").title()
    pages = _split_pages(raw)
    chunks = _split_chunks(doc_id, title, pages)

    counts: dict[str, int] = {}
    for chunk in chunks:
        for token in chunk.tokens:
            if len(token) > 4:
                counts[token] = counts.get(token, 0) + 1
    keywords = [w for w, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:6]]

    return LoadedDoc(
        doc_id=doc_id,
        path=path,
        title=title,
        ext=path.suffix,
        mime=MIME_BY_EXT.get(path.suffix, "application/octet-stream"),
        size_bytes=stat.st_size,
        mtime=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
        pages=pages,
        chunks=chunks,
        keywords=keywords,
    )


def load_corpus(directory: Path) -> list[LoadedDoc]:
    files = sorted(
        p for p in directory.rglob("*") if p.is_file() and p.suffix in {".md", ".txt"}
    )
    return [load_document(p) for p in files]
