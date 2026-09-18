"""Parsing to pages. Page numbers are a correctness requirement: citations carry them."""

from __future__ import annotations

from pathlib import Path

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
