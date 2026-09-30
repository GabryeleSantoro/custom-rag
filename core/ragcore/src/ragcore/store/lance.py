"""LanceDB tables: chunks with their vectors, plus index_meta.

`documents` lives in SQLite rather than here. LanceDB holds what needs vector
or full-text search; everything else is relational state the UI pages through.

The `parents` table from the build plan is deliberately absent: naive chunking
produces no parent/child relationship, so context packing works on children.
It arrives with real chunking.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import TypedDict

import lancedb
import pyarrow as pa

from ragcore.models.embed import EMBED_DIM

CHUNKS = "chunks"
META = "index_meta"


class ChunkRow(TypedDict):
    chunk_id: str
    doc_id: str
    doc_title: str
    page_start: int
    page_end: int
    section_path: str
    text: str
    embed_text: str
    n_tokens: int
    vector: list[float]


CHUNK_SCHEMA = pa.schema(
    [
        ("chunk_id", pa.string()),
        ("doc_id", pa.string()),
        ("doc_title", pa.string()),
        ("page_start", pa.int32()),
        ("page_end", pa.int32()),
        ("section_path", pa.string()),
        ("text", pa.string()),
        ("embed_text", pa.string()),
        ("n_tokens", pa.int32()),
        ("vector", pa.list_(pa.float32(), EMBED_DIM)),
    ]
)

META_SCHEMA = pa.schema([("key", pa.string()), ("value", pa.string())])


def _sql_list(values: list[str]) -> str:
    """Quote values for an SQL ``IN (...)`` predicate; a ``'`` inside is doubled."""
    return ", ".join("'" + value.replace("'", "''") + "'" for value in values)


class VectorStore:
    """Chunk vectors/text and the index_meta guard, both backed by LanceDB tables.

    One `VectorStore` owns one on-disk LanceDB database directory (`root`), which
    holds exactly the two tables above. Re-opening the same `root` reattaches to
    the same tables rather than recreating them.
    """

    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(root))
        self.chunks = self._table(CHUNKS, CHUNK_SCHEMA)
        self.meta = self._table(META, META_SCHEMA)

    def _table(self, name: str, schema: pa.Schema):
        # `table_names()` is deprecated in favour of `list_tables()` as of lancedb 0.39.
        if name in self.db.list_tables().tables:
            return self.db.open_table(name)
        return self.db.create_table(name, schema=schema)

    def add_chunks(self, rows: list[ChunkRow]) -> None:
        if rows:
            self.chunks.add(rows)

    def delete_by_doc(self, doc_ids: list[str]) -> None:
        """Delete every chunk belonging to any of `doc_ids`.

        An empty list is a deliberate no-op, not "delete everything": LanceDB's
        `delete` takes a SQL predicate, and an empty `IN (...)` would either error
        or (worse) be misread as unconditional, so it is guarded here.
        """
        if not doc_ids:
            return
        self.chunks.delete(f"doc_id IN ({_sql_list(doc_ids)})")

    def clear(self) -> None:
        self.chunks.delete("true")

    def optimize(self) -> None:
        """Merge the small fragments each write leaves and drop superseded versions."""
        self.chunks.optimize(cleanup_older_than=timedelta(0))

    def dense(
        self, vector: list[float], k: int, *, doc_ids: list[str] | None = None
    ) -> list[tuple[str, float]]:
        """Nearest chunks by cosine similarity, best first.

        LanceDB's `_distance` under the "cosine" metric is `1 - cosine_similarity`;
        this converts back to similarity before returning. `k` larger than the row
        count, or an empty table, simply yields however many rows exist (LanceDB
        clamps the limit itself; no error). `doc_ids` restricts the search before
        the top-k cut, so a scoped query still gets `k` hits from its documents.
        """
        query = self.chunks.search(vector, vector_column_name="vector").metric("cosine")
        if doc_ids is not None:
            query = query.where(f"doc_id IN ({_sql_list(doc_ids)})", prefilter=True)
        hits = query.limit(k).select(["chunk_id", "_distance"]).to_list()
        # LanceDB returns cosine *distance*; the pipeline wants similarity.
        return [(hit["chunk_id"], 1.0 - float(hit["_distance"])) for hit in hits]

    def fts(self, query: str, k: int) -> list[tuple[str, float]]:
        raise NotImplementedError("full-text search lands in Task 12")

    def get(self, chunk_ids: list[str]) -> list[ChunkRow]:
        """Fetch rows by id, returned in the order `chunk_ids` was given.

        Ids absent from the table are silently dropped rather than raising, so
        the result can be shorter than the input but never reordered relative to
        it or padded with placeholders.
        """
        if not chunk_ids:
            return []
        found = (
            self.chunks.search()
            .where(f"chunk_id IN ({_sql_list(chunk_ids)})")
            .limit(len(chunk_ids))
            .to_list()
        )
        by_id = {row["chunk_id"]: row for row in found}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def chunk_count(self) -> int:
        return self.chunks.count_rows()

    def read_meta(self) -> dict[str, str]:
        return {row["key"]: row["value"] for row in self.meta.search().limit(10_000).to_list()}

    def write_meta(self, values: dict[str, str]) -> None:
        if values:
            self.meta.delete(f"key IN ({_sql_list(list(values))})")
        self.meta.add([{"key": key, "value": value} for key, value in values.items()])
