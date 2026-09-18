"""Finding files worth indexing, and fingerprinting them.

The digest is what makes re-indexing a no-op: a file whose sha256 is unchanged
is skipped without being parsed or embedded (Task 9 checks it against
``MetaStore.sha_index`` before doing either).

Semantics later tasks (8, 9, 14) rely on, decided here deliberately:

- **Globs are matched relative to ``root``.** ``include_globs``/``exclude_globs``
  are passed straight to ``Path.glob`` on ``root``, so a pattern like
  ``"**/*.md"`` is interpreted relative to ``root`` and never sees an absolute
  path. Task 9 forwards user-supplied globs unchanged, so this is the contract
  a caller's glob strings must be written against.
- **Exclude always wins.** A file is kept only if it matches at least one
  ``include_globs`` pattern *and* matches none of the ``exclude_globs``
  patterns. A file matching both is dropped — excludes are a hard veto, not a
  second vote.
- **``max_file_mb`` is mebibytes** (``max_file_mb * 1024 * 1024`` bytes), and
  the cap is inclusive: a file of exactly that many bytes is kept; only a file
  *strictly larger* is skipped.
- **Hidden files and directories are always skipped**, regardless of
  ``include_globs``. ``pathlib.Path.glob`` matches dotfiles by default (unlike
  a shell), so a naive ``"**/*.md"`` would otherwise descend into things like
  ``.git``, ``.obsidian`` or ``.venv``. Any path whose name, or an ancestor
  directory's name (relative to ``root``), starts with ``"."`` is excluded
  unconditionally — a caller cannot opt back in via ``include_globs``.
- **Symlinked files are indexed like any other file; symlinked directories are
  never traversed.** On Python >=3.13 (this repo's floor), ``Path.glob``'s
  recursive ``**`` defaults to ``recurse_symlinks=False``, so a symlink cycle
  under ``root`` cannot make the walk recurse forever. We rely on that default
  rather than re-implementing cycle detection.
- **``mime`` comes from a small static extension table, not the stdlib
  ``mimetypes`` module.** ``mimetypes.guess_type`` reads the host's installed
  mime database, which varies by platform and can produce surprising results
  for uncommon extensions (e.g. ``.xyz`` resolves to ``chemical/x-xyz`` on this
  machine). A fixed table keeps ``mime`` deterministic across every machine
  this runs on; an extension outside the table gets
  ``"application/octet-stream"``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

MIME_BY_EXT = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".pdf": "application/pdf",
}

READ_CHUNK = 1024 * 1024  # stream digests in 1 MiB blocks; never read a whole file into memory


@dataclass(slots=True)
class FoundFile:
    path: Path
    sha256: str
    size_bytes: int
    mtime: datetime
    ext: str
    mime: str


def sha256_of(path: Path) -> str:
    """Digest a file's bytes, streaming so a large file is never read whole.

    The digest depends only on content, not on how the file happens to be
    chunked while reading it, so it is stable across file sizes and re-runs.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(READ_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _is_hidden(relative: Path) -> bool:
    """True if any path segment (relative to the walk root) starts with a dot."""
    return any(part.startswith(".") for part in relative.parts)


def walk_source(
    root: Path,
    *,
    include_globs: list[str],
    exclude_globs: list[str],
    max_file_mb: int,
) -> list[FoundFile]:
    """Find files under ``root`` worth indexing.

    See the module docstring for the exact glob, hidden-file, symlink and size
    rules this applies. Results are sorted by path for a deterministic order.
    """
    cap_bytes = max_file_mb * 1024 * 1024

    included: set[Path] = set()
    for pattern in include_globs:
        included.update(p for p in root.glob(pattern) if p.is_file())
    for pattern in exclude_globs:
        included.difference_update(root.glob(pattern))

    out: list[FoundFile] = []
    for path in sorted(included):
        if _is_hidden(path.relative_to(root)):
            continue
        stat = path.stat()
        if stat.st_size > cap_bytes:
            continue
        ext = path.suffix.lower()
        out.append(
            FoundFile(
                path=path,
                sha256=sha256_of(path),
                size_bytes=stat.st_size,
                mtime=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
                ext=ext,
                mime=MIME_BY_EXT.get(ext, "application/octet-stream"),
            )
        )
    return out
