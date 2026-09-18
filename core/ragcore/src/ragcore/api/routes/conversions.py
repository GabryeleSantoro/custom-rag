"""Research-backed conversion of indexed slide documents into Markdown."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx
from fastapi import APIRouter, HTTPException

from ragcore.api.deps import AnswererDep, ConfigDep, StoreDep
from ragcore.api.schemas import (
    ConversionDoneEvent,
    ConversionResearchEvent,
    ConversionSavedEvent,
    ConversionStartEvent,
    RetrievedChunk,
    SlideConversionRequest,
    SourceCreate,
    WebResearchResult,
)
from ragcore.api.sse import frame, sse_response
from ragcore.ingest.parse import UnsupportedFormat, parse

router = APIRouter(prefix="/conversions", tags=["conversions"])


class _SearchParser(HTMLParser):
    """Small dependency-free parser for DuckDuckGo's HTML result page."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[WebResearchResult] = []
        self._current: dict[str, str] | None = None
        self._part: str | None = None
        self._body_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "div" and "result__body" in classes:
            self._current = {}
            self._body_depth = 1
            return
        if self._current is None:
            return
        if tag == "div":
            self._body_depth += 1
        if tag == "a" and "result__a" in classes:
            href = html.unescape(attributes.get("href") or "")
            parsed = urlparse(href if not href.startswith("//") else f"https:{href}")
            redirect = parse_qs(parsed.query).get("uddg", [""])[0]
            self._current["url"] = unquote(redirect or href)
            self._part = "title"
        elif "result__snippet" in classes:
            self._part = "snippet"

    def handle_data(self, data: str) -> None:
        if self._current is not None and self._part:
            key = self._part
            self._current[key] = f"{self._current.get(key, '')}{data}"

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._part = None
        if tag == "div" and self._current:
            self._body_depth -= 1
        if tag == "div" and self._body_depth == 0 and self._current and self._current.get("title"):
            url = self._current.get("url", "")
            if url and url.startswith("http"):
                self.results.append(
                    WebResearchResult(
                        title=" ".join(self._current["title"].split()),
                        url=url,
                        snippet=" ".join(self._current.get("snippet", "").split()),
                    )
                )
            self._current = None


async def _search_web(query: str) -> tuple[list[WebResearchResult], str | None]:
    """Return a few current web references without adding a search SDK dependency."""

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(12.0, connect=5.0),
            headers={"User-Agent": "custom-rag/0.1 research converter"},
            follow_redirects=True,
        ) as client:
            response = await client.get(f"https://html.duckduckgo.com/html/?q={quote_plus(query)}")
            response.raise_for_status()
    except httpx.HTTPError as exc:
        return [], f"Ricerca web non disponibile: {type(exc).__name__}"

    parser = _SearchParser()
    parser.feed(response.text)
    unique: list[WebResearchResult] = []
    seen: set[str] = set()
    for result in parser.results:
        if result.url in seen:
            continue
        seen.add(result.url)
        unique.append(result)
        if len(unique) == 6:
            break
    return unique, None if unique else "Nessun risultato web trovato per questa ricerca."


def _slug(value: str) -> str:
    value = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE).strip().lower()
    value = re.sub(r"[-\s]+", "-", value)
    return value[:72] or "conversione-slide"


def _clean_markdown(text: str, title: str, results: list[WebResearchResult]) -> str:
    """Keep the model output readable and make web provenance durable in the file."""

    cleaned = text.strip().strip("`").strip()
    if not cleaned.startswith("#"):
        cleaned = f"# {title}\n\n{cleaned}"
    if results:
        sources = ["", "## Fonti web", ""]
        sources.extend(f"- [{result.title}]({result.url}) — {result.snippet}" for result in results)
        cleaned = f"{cleaned.rstrip()}\n" + "\n".join(sources)
    return cleaned + "\n"


def _global_source(store, global_dir: Path):
    for source in store.sources.values():
        if source.project_id is None and Path(source.path) == global_dir:
            return source
    return store.add_source(
        SourceCreate(
            path=str(global_dir),
            include_globs=["**/*.md"],
            exclude_globs=[],
            max_file_mb=100,
            watch=True,
            project_id=None,
        )
    )


def _chunks_from_pages(
    doc_id: str,
    title: str,
    pages: list[tuple[int, str | None, str]],
) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=f"{doc_id}#slide-{page_number}",
            doc_id=doc_id,
            doc_title=title,
            page_start=page_number,
            page_end=page_number,
            section_path=section_path,
            text=text[:7000],
        )
        for page_number, section_path, text in pages
        if text.strip()
    ]


@router.post("/slides")
async def convert_slides(
    payload: SlideConversionRequest,
    config: ConfigDep,
    store: StoreDep,
    answerer: AnswererDep,
):
    active = next(
        (connection for connection in store.connections.values() if connection.active), None
    )
    if active is None:
        raise HTTPException(409, "Connect a generation model before converting slides")

    pages: list[RetrievedChunk] = []
    input_titles: list[str] = []
    if payload.slide_ids:
        documents = [store.documents.get(doc_id) for doc_id in payload.slide_ids]
        if any(document is None for document in documents):
            raise HTTPException(404, "One or more selected slide files were not found")
        for document in (item for item in documents if item is not None):
            content = store.content(document.id)
            if content is None:
                raise HTTPException(422, f"No readable content for {document.title}")
            input_titles.append(document.title)
            pages.extend(
                _chunks_from_pages(
                    document.id,
                    document.title,
                    [(page.page, page.section_path, page.text) for page in content.pages],
                )
            )

    for index, file_path in enumerate(payload.file_paths):
        path = Path(file_path).expanduser()
        if not path.is_file():
            raise HTTPException(404, f"Slide file not found: {file_path}")
        try:
            parsed = parse(path)
        except (OSError, UnsupportedFormat, ValueError) as exc:
            raise HTTPException(422, f"Could not read slide file {path.name}: {exc}") from exc
        title_for_file = parsed.title or path.stem
        input_titles.append(title_for_file)
        pages.extend(
            _chunks_from_pages(
                f"upload-{index}",
                title_for_file,
                [(page.page, page.section_path, page.text) for page in parsed.pages],
            )
        )

    if not pages:
        raise HTTPException(422, "The selected files do not contain any slide text")

    title = payload.output_title or input_titles[0]
    query = payload.research_query or (
        f"{title}: key concepts, current context, examples and sources"
    )
    results, warning = await _search_web(query)
    web_chunks = [
        RetrievedChunk(
            chunk_id=f"web-{index}",
            doc_id=f"web-{index}",
            doc_title=result.title,
            page_start=1,
            page_end=1,
            text=f"{result.title}. {result.snippet} Source: {result.url}",
        )
        for index, result in enumerate(results)
    ]
    context = pages[:36] + web_chunks
    instruction = (
        f"Trasforma le slide in un testo compiuto e approfondito in "
        f"{'italiano' if payload.language == 'it' else 'inglese'}. "
        "Scrivi direttamente Markdown valido, senza delimitatori ``` e senza parlare del processo. "
        "Mantieni i concetti delle slide, collega le idee in una narrazione leggibile, "
        "aggiungi definizioni, contesto, esempi e implicazioni usando la ricerca web fornita. "
        "Distingui chiaramente fatti e inferenze. "
        f"Livello di approfondimento: {'alto' if payload.depth == 'deep' else 'standard'}. "
        "Usa un titolo H1, sezioni H2/H3, paragrafi e liste quando aiutano."
    )

    async def events():
        yield frame(
            "conversion_start",
            ConversionStartEvent(slide_ids=payload.slide_ids, title=title),
        )
        yield frame(
            "conversion_research",
            ConversionResearchEvent(query=query, results=results, warning=warning),
        )
        output: list[str] = []
        try:
            async for piece in answerer.stream(instruction, context, set()):
                output.append(piece)
                yield frame("token", {"text": piece})
        except Exception as exc:  # noqa: BLE001 - reported as an SSE error
            yield frame("error", {"message": str(exc), "retryable": True})
            return

        markdown = _clean_markdown("".join(output), title, results)
        global_dir = config.data_dir / "global-files"
        global_dir.mkdir(parents=True, exist_ok=True)
        output_path = global_dir / f"{_slug(title)}.md"
        output_path.write_text(markdown, encoding="utf-8")

        source = _global_source(store, global_dir)
        indexed = store.ingest_source(source.id)
        saved = next((document for document in indexed if Path(document.path) == output_path), None)
        if saved is None:
            yield frame(
                "error",
                {
                    "message": "Markdown file was saved but could not be indexed",
                    "retryable": False,
                },
            )
            return

        yield frame(
            "conversion_saved",
            ConversionSavedEvent(path=str(output_path), title=saved.title, document_id=saved.id),
        )
        yield frame(
            "conversion_done",
            ConversionDoneEvent(
                path=str(output_path),
                title=saved.title,
                document_id=saved.id,
                research_count=len(results),
            ),
        )

    return sse_response(events())
