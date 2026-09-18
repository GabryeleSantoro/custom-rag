"""Parsing to pages. Page numbers are a correctness requirement: citations carry them."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from ragcore.ingest.parse import UnsupportedFormat, parse


def test_markdown_splits_into_pages_on_level_two_headings(tmp_path: Path) -> None:
    source = tmp_path / "reranking.md"
    source.write_text(
        "# Reranking\n\nIntro text.\n\n"
        "## Cross encoders\n\nThey score pairs.\n\n"
        "## Latency\n\nThey are slower.\n"
    )

    parsed = parse(source)

    assert parsed.title == "Reranking"
    assert [p.page for p in parsed.pages] == [1, 2, 3]
    assert parsed.pages[1].section_path == "Reranking > Cross encoders"
    assert "score pairs" in parsed.pages[1].text


def test_markdown_without_headings_is_one_page(tmp_path: Path) -> None:
    source = tmp_path / "flat.md"
    source.write_text("Just a paragraph with no headings at all.")

    parsed = parse(source)

    assert [p.page for p in parsed.pages] == [1]
    assert parsed.title == "flat"


def test_plain_text_is_one_page(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("Chunking splits documents into retrievable units.")

    parsed = parse(source)

    assert len(parsed.pages) == 1
    assert parsed.needs_ocr is False


def test_unknown_extension_raises(tmp_path: Path) -> None:
    source = tmp_path / "thing.xyz"
    source.write_text("?")

    with pytest.raises(UnsupportedFormat):
        parse(source)


def test_pdf_extracts_slide_text_page_by_page(tmp_path: Path, monkeypatch) -> None:
    class FakeTextPage:
        def __init__(self, text: str) -> None:
            self.text = text

        def get_text_range(self) -> str:
            return self.text

        def close(self) -> None:
            pass

    class FakePage:
        def __init__(self, text: str) -> None:
            self.text = text

        def get_textpage(self) -> FakeTextPage:
            return FakeTextPage(self.text)

        def close(self) -> None:
            pass

    class FakeDocument:
        def __init__(self, _path: str) -> None:
            self.pages = [FakePage("Slide title\r\nElaborated slide text."), FakePage("")]

        def __len__(self) -> int:
            return len(self.pages)

        def __getitem__(self, index: int) -> FakePage:
            return self.pages[index]

        def close(self) -> None:
            pass

    monkeypatch.setitem(sys.modules, "pypdfium2", SimpleNamespace(PdfDocument=FakeDocument))
    source = tmp_path / "presentation.pdf"
    source.write_bytes(b"not decoded as UTF-8")

    parsed = parse(source)

    assert parsed.title == "presentation"
    assert len(parsed.pages) == 1
    assert parsed.pages[0].text == "Slide title\nElaborated slide text."
    assert parsed.needs_ocr is True


def test_pptx_extracts_visible_text_in_slide_order(tmp_path: Path) -> None:
    source = tmp_path / "presentation.pptx"
    slide = (
        '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        "<a:t>Second slide content</a:t></p:sld>"
    )
    with ZipFile(source, "w") as archive:
        archive.writestr("ppt/slides/slide2.xml", slide)
        archive.writestr("ppt/slides/slide1.xml", slide.replace("Second", "First"))

    parsed = parse(source)

    assert [page.page for page in parsed.pages] == [1, 2]
    assert parsed.pages[0].text == "First slide content"
    assert parsed.pages[1].text == "Second slide content"


def test_pptx_appends_speaker_notes_when_present(tmp_path: Path) -> None:
    source = tmp_path / "presentation.pptx"
    slide = (
        '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        "<a:t>Slide content</a:t></p:sld>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" '
        'Target="../notesSlides/notesSlide1.xml"/>'
        "</Relationships>"
    )
    notes = (
        '<p:notes xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        "<a:t>Remember to mention the Q3 numbers.</a:t></p:notes>"
    )
    with ZipFile(source, "w") as archive:
        archive.writestr("ppt/slides/slide1.xml", slide)
        archive.writestr("ppt/slides/_rels/slide1.xml.rels", rels)
        archive.writestr("ppt/notesSlides/notesSlide1.xml", notes)

    parsed = parse(source)

    assert parsed.pages[0].text == (
        "Slide content\n\nNote del relatore: Remember to mention the Q3 numbers."
    )


def test_pptx_appends_external_hyperlinks_when_present(tmp_path: Path) -> None:
    source = tmp_path / "presentation.pptx"
    slide = (
        '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<a:r><a:rPr><a:hlinkClick r:id=\"rId2\"/></a:rPr><a:t>Learn more</a:t></a:r>"
        "</p:sld>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" '
        'Target="https://example.com/reference" TargetMode="External"/>'
        "</Relationships>"
    )
    with ZipFile(source, "w") as archive:
        archive.writestr("ppt/slides/slide1.xml", slide)
        archive.writestr("ppt/slides/_rels/slide1.xml.rels", rels)

    parsed = parse(source)

    assert parsed.pages[0].text == (
        "Learn more\n\nLink nella slide: https://example.com/reference"
    )


def test_pptx_slide_without_rels_file_is_unaffected(tmp_path: Path) -> None:
    source = tmp_path / "presentation.pptx"
    slide = (
        '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        "<a:t>Plain slide</a:t></p:sld>"
    )
    with ZipFile(source, "w") as archive:
        archive.writestr("ppt/slides/slide1.xml", slide)

    parsed = parse(source)

    assert parsed.pages[0].text == "Plain slide"
