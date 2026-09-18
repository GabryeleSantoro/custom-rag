"""File to pages.

A "page" is whatever unit a citation can point at. For PDFs (Task 14) that is
a real page; for markdown/text it is a level-two section, which is the
closest honest analogue and is what the reader view highlights against. Task
8 chunks within a page's text; it never crosses a page boundary.

``parse()`` dispatches on suffix through the ``_PARSERS`` table below. PDFs
are opened with ``pypdfium2`` rather than being decoded as UTF-8: a PDF is a
binary container, and reading it as text returns font tables and drawing
operators instead of the words visible on the slide. ``ParsedDoc.needs_ocr``
is set when a PDF page has no extractable text layer (for example, a slide
exported as an image).

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
import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

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
            ((section_path, body) for section_path, body in sections if body), start=1
        )
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


def _clean_pdf_text(text: str) -> str:
    """Normalize text returned by PDFium without changing its character offsets."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def _pdf(path: Path) -> ParsedDoc:
    """Extract one page per PDF page using PDFium's text layer.

    OCR is intentionally not run here. Pages without a text layer are omitted
    from the searchable text and flagged so the ingestion job can send them
    through the OCR stage once that stage is enabled.
    """
    try:
        import pypdfium2 as pdfium
    except ModuleNotFoundError as exc:  # pragma: no cover - packaging guard
        raise RuntimeError(
            "PDF support requires pypdfium2; reinstall the ragcore dependencies"
        ) from exc

    document = pdfium.PdfDocument(str(path))
    sections: list[tuple[str, str]] = []
    needs_ocr = False
    try:
        for page_number in range(len(document)):
            page = document[page_number]
            text_page = None
            try:
                text_page = page.get_textpage()
                text = _clean_pdf_text(text_page.get_text_range())
            finally:
                if text_page is not None:
                    text_page.close()
                page.close()

            if text:
                sections.append((f"{path.stem} > Page {page_number + 1}", text))
            else:
                needs_ocr = True
    finally:
        document.close()

    return ParsedDoc(title=path.stem, pages=_paginate(sections), needs_ocr=needs_ocr)


def _pptx(path: Path) -> ParsedDoc:
    """Extract visible text from a PowerPoint package, one page per slide."""
    with ZipFile(path) as archive:
        slide_names = sorted(
            name
            for name in archive.namelist()
            if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
        )
        slide_names.sort(key=lambda name: int(re.search(r"slide(\d+)\.xml$", name).group(1)))
        sections: list[tuple[str, str]] = []
        for slide_number, name in enumerate(slide_names, start=1):
            root = ET.fromstring(archive.read(name))
            text = " ".join(
                node.text.strip()
                for node in root.iter()
                if node.tag.endswith("}t") and node.text and node.text.strip()
            )
            sections.append((f"{path.stem} > Slide {slide_number}", text))
    return ParsedDoc(title=path.stem, pages=_paginate(sections))


_PARSERS: dict[str, Callable[[Path], ParsedDoc]] = {
    ".md": _markdown,
    ".txt": _plain,
    ".pdf": _pdf,
    ".pptx": _pptx,
}


def parse(path: Path) -> ParsedDoc:
    """Parse ``path`` into a ``ParsedDoc``, dispatching on its lowercased suffix."""
    handler = _PARSERS.get(path.suffix.lower())
    if handler is None:
        raise UnsupportedFormat(f"no parser for {path.suffix!r}")
    return handler(path)
