"""Retrieval: dense now, plus BM25 fusion and reranking in later tasks."""

from __future__ import annotations

import time

from ragcore.api.schemas import (
    QueryFilters,
    RetrievalSettings,
    RetrievedChunk,
    StageLatency,
)
from ragcore.retrieve.pack import pack
from ragcore.store.lance import ChunkRow, VectorStore
from ragcore.stub.retrieval import allowed


def _to_chunk(row: ChunkRow, *, dense_rank: int | None, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        doc_title=row["doc_title"],
        page_start=int(row["page_start"]),
        page_end=int(row["page_end"]),
        section_path=row["section_path"] or None,
        text=row["text"],
        dense_rank=dense_rank,
        rerank_score=score,
    )


class HybridRetriever:
    def __init__(self, vectors: VectorStore, embedder, *, reranker=None) -> None:
        self.vectors = vectors
        self.embedder = embedder
        self.reranker = reranker

    async def asearch(
        self,
        query: str,
        *,
        settings: RetrievalSettings,
        filters: QueryFilters,
        doc_meta: dict[str, dict],
    ) -> tuple[list[RetrievedChunk], StageLatency, int]:
        latency = StageLatency()
        docs = {doc_id for doc_id in doc_meta if allowed(doc_id, filters, doc_meta)}
        if not query.strip() or not docs:
            return [], latency, 0

        t0 = time.perf_counter()
        [vector] = await self.embedder.embed([query], query=True)
        latency.embed_ms = (time.perf_counter() - t0) * 1000

        # An unfiltered query skips the IN list; chunks of unlisted documents
        # (a write the library never saw) are dropped below instead.
        # ponytail: LanceDB calls block the event loop; to_thread them if a big
        # index makes queries stall ingest progress.
        t0 = time.perf_counter()
        scoped = sorted(docs) if len(docs) < len(doc_meta) else None
        hits = self.vectors.dense(vector, k=settings.dense_top_k, doc_ids=scoped)
        latency.dense_ms = (time.perf_counter() - t0) * 1000

        rows = {row["chunk_id"]: row for row in self.vectors.get([cid for cid, _ in hits])}
        scored = [
            _to_chunk(rows[cid], dense_rank=rank + 1, score=score)
            for rank, (cid, score) in enumerate(hits)
            if cid in rows and rows[cid]["doc_id"] in docs
        ]
        candidates = len(scored)

        # `min_score` is calibrated for the reranker's sigmoid probabilities;
        # cosine and RRF scores sit near 0.0-0.05 and a 0.3 gate would drop
        # everything. It belongs to the rerank stage (Task 13).
        t0 = time.perf_counter()
        kept = pack(scored[: settings.top_k], settings.context_token_budget)
        latency.pack_ms = (time.perf_counter() - t0) * 1000
        latency.total_ms = latency.embed_ms + latency.dense_ms + latency.pack_ms
        return kept, latency, candidates
