"""Walking a source folder: globs, size cap, stable digests."""

from __future__ import annotations

from pathlib import Path

from ragcore.ingest.walk import sha256_of, walk_source


def build_corpus(root: Path) -> None:
    (root / "notes").mkdir(parents=True)
    (root / "notes" / "reranking.md").write_text("# Reranking\n\nCross-encoders reorder.")
    (root / "notes" / "chunking.txt").write_text("Chunking splits documents.")
    (root / "notes" / "ignore.log").write_text("noise")
    (root / "notes" / "huge.md").write_text("x" * (2 * 1024 * 1024))


def test_include_globs_select_files(tmp_path: Path) -> None:
    build_corpus(tmp_path)

    found = walk_source(tmp_path, include_globs=["**/*.md"], exclude_globs=[], max_file_mb=100)

    assert sorted(f.path.name for f in found) == ["huge.md", "reranking.md"]


def test_exclude_globs_win(tmp_path: Path) -> None:
    build_corpus(tmp_path)

    found = walk_source(
        tmp_path, include_globs=["**/*"], exclude_globs=["**/*.log", "**/huge.md"], max_file_mb=100
    )

    assert sorted(f.path.name for f in found) == ["chunking.txt", "reranking.md"]


def test_size_cap_skips_large_files(tmp_path: Path) -> None:
    build_corpus(tmp_path)

    found = walk_source(tmp_path, include_globs=["**/*.md"], exclude_globs=[], max_file_mb=1)

    assert [f.path.name for f in found] == ["reranking.md"]


def test_digest_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    build_corpus(tmp_path)
    target = tmp_path / "notes" / "reranking.md"
    before = sha256_of(target)
    assert before == sha256_of(target)

    target.write_text("# Reranking\n\nChanged.")

    assert sha256_of(target) != before


def test_mime_is_derived_from_the_extension(tmp_path: Path) -> None:
    build_corpus(tmp_path)

    found = walk_source(tmp_path, include_globs=["**/*.txt"], exclude_globs=[], max_file_mb=100)

    assert found[0].mime == "text/plain"
