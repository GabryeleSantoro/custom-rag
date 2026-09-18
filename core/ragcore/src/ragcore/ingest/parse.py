"""File to pages.

A "page" is whatever unit a citation can point at. For PDFs (Task 14) that is
a real page; for markdown/text it is a level-two section, which is the
closest honest analogue and is what the reader view highlights against. Task
8 chunks within a page's text; it never crosses a page boundary.

``parse()`` dispatches on suffix through the ``_PARSERS`` table below, so
Task 14 adding PDF support is one new entry (``".pdf": _pdf``) plus a
function — not a change to ``parse()`` itself. ``ParsedDoc.needs_ocr``
exists for that future case (a PDF with no extractable text layer); nothing
this task parses can ever set it True, and both handlers below leave it at
its default of ``False``.

Semantics decided here, that Task 8/9 depend on:

- **``section_path`` separator is ``" > "``**, most-general first ((document
  title) > (heading)). It is one segment deep for now because we only split
  on level-two (``##``) headings; deeper headings (``###`` and beyond) stay
  inside their enclosing section's text rather than creating more pages.
- **Text before the first level-two heading** (including all of a level-one
  title line) becomes page 1, with ``section_path`` equal to the document
  title alone (no ``" > "`` suffix) — it is the document's own top-level
  section, not a subsection of itself. If there is no such preamble text
  (the document starts directly at a ``##`` heading), no page is created for
  it.
- **The title** is the first level-one (``#``) heading's text if the document
  has one, otherwise the filename stem. This applies to both markdown and
  plain text (plain text never has a level-one heading, so it always falls
  back to the stem).
- **Encoding**: files are read as UTF-8 with ``errors="replace"``. Real
  corpora contain files that are not valid UTF-8 (legacy exports, mixed
  encodings); refusing to ingest them, or crashing the whole walk on one bad
  file, is worse than replacing the undecodable bytes with U+FFFD and
  indexing what does decode. A file that is mostly non-UTF-8 will simply
  chunk and embed poorly, which is a quality problem, not a crash.
- **A page with empty text (after stripping) is dropped, not returned.** An
  empty file, or a markdown file that is nothing but headings with no body
  under any of them, would otherwise produce one or more pages whose ``text``
  is ``""``. Task 8's chunker already skips those, but a document that
  reports ``n_pages > 0`` while producing zero chunks looks indexed when it
  is not. Dropping empty pages here means a document with no usable text
  comes back with ``pages == []``, which Task 9 can record honestly instead
  of silently pretending something was indexed. The kept pages are
  renumbered ``1..n`` contiguously — a dropped page never leaves a gap in the
  sequence.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_HEADING = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)


class UnsupportedFormat(Exception):
    """Raised by ``parse()`` when no handler is registered for a file's suffix."""


@dataclass(slots=True)
class ParsedPage:
    page: int
    section_path: str
    text: str


@dataclass(slots=True)
class ParsedDoc:
    title: str
    pages: list[ParsedPage]
    needs_ocr: bool = False


def _title(text: str, fallback: str) -> str:
    match = next((m for m in _HEADING.finditer(text) if len(m.group(1)) == 1), None)
    return match.group(2).strip() if match else fallback


def _paginate(sections: list[tuple[str, str]]) -> list[ParsedPage]:
    """Number non-empty sections 1..n; a section with empty text is dropped."""
    return [
        ParsedPage(page=i, section_path=section_path, text=body)
        for i, (section_path, body) in enumerate(
            (sp, b) for sp, b in sections if b
        , start=1)
    ]


def _markdown(path: Path) -> ParsedDoc:
    text = path.read_text(encoding="utf-8", errors="replace")
    title = _title(text, path.stem)
    marks = [m for m in _HEADING.finditer(text) if len(m.group(1)) == 2]

    if not marks:
        return ParsedDoc(title=title, pages=_paginate([(title, text.strip())]))

    sections: list[tuple[str, str]] = [(title, text[: marks[0].start()].strip())]
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        heading = mark.group(2).strip()
        sections.append((f"{title} > {heading}", text[mark.end() : end].strip()))

    return ParsedDoc(title=title, pages=_paginate(sections))


def _plain(path: Path) -> ParsedDoc:
    text = path.read_text(encoding="utf-8", errors="replace")
    return ParsedDoc(title=path.stem, pages=_paginate([(path.stem, text.strip())]))


_PARSERS: dict[str, Callable[[Path], ParsedDoc]] = {
    ".md": _markdown,
    ".txt": _plain,
}


def parse(path: Path) -> ParsedDoc:
    """Parse ``path`` into a ``ParsedDoc``, dispatching on its lowercased suffix."""
    handler = _PARSERS.get(path.suffix.lower())
    if handler is None:
        raise UnsupportedFormat(f"no parser for {path.suffix!r}")
    return handler(path)
