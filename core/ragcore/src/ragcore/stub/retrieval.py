"""A real retrieval pipeline over fixture text.

BM25 stands in for the keyword leg and a TF-IDF cosine stands in for the dense
leg. Both are genuine — they return different orderings for the same query,
which is the property the fusion step exists to exploit, and it means the UI
shows honest ranks and scores instead of invented ones.
"""

from __future__ import annotations

import math
import time
from collections import Counter

from ragcore.api.schemas import QueryFilters, RetrievalSettings, RetrievedChunk, StageLatency
from ragcore.stub.corpus import Chunk, tokenize

K1 = 1.5
B = 0.75


class Retriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.doc_freq: Counter[str] = Counter()
        for chunk in chunks:
            self.doc_freq.update(set(chunk.tokens))
        self.n = max(len(chunks), 1)
        self.avg_len = sum(len(c.tokens) for c in chunks) / self.n if chunks else 1.0
        self._tfidf = [self._vector(c.tokens) for c in chunks]

    def _idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def _vector(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        vec = {t: (1 + math.log(c)) * self._idf(t) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def _bm25(self, query: list[str], index: int) -> float:
        chunk = self.chunks[index]
        counts = Counter(chunk.tokens)
        length = len(chunk.tokens) or 1
        score = 0.0
        for term in query:
            tf = counts.get(term, 0)
            if not tf:
                continue
            denom = tf + K1 * (1 - B + B * length / self.avg_len)
            score += self._idf(term) * (tf * (K1 + 1)) / denom
        return score

    def _dense(self, query_vec: dict[str, float], index: int) -> float:
        vec = self._tfidf[index]
        return sum(weight * vec.get(term, 0.0) for term, weight in query_vec.items())

    def _allowed(self, chunk: Chunk, filters: QueryFilters, doc_meta: dict[str, dict]) -> bool:
        meta = doc_meta.get(chunk.doc_id)
        if meta is None:
            return False
        if filters.doc_ids and chunk.doc_id not in filters.doc_ids:
            return False
        if filters.source_ids and meta["source_id"] not in filters.source_ids:
            return False
        if filters.exts and meta["ext"] not in filters.exts:
            return False
        if filters.langs and meta.get("lang") not in filters.langs:
            return False
        if filters.after and meta["mtime"] < filters.after:
            return False
        return not (filters.before and meta["mtime"] > filters.before)

    def search(
        self,
        query: str,
        *,
        settings: RetrievalSettings,
        filters: QueryFilters,
        doc_meta: dict[str, dict],
    ) -> tuple[list[RetrievedChunk], StageLatency, int]:
        """Dense + BM25 → reciprocal rank fusion → rerank → min-score gate."""
        latency = StageLatency()
        terms = tokenize(query)
        if not terms or not self.chunks:
            return [], latency, 0

        t0 = time.perf_counter()
        query_vec = self._vector(terms)
        latency.embed_ms = (time.perf_counter() - t0) * 1000

        candidates = [
            i for i, c in enumerate(self.chunks) if self._allowed(c, filters, doc_meta)
        ]

        t0 = time.perf_counter()
        dense = sorted(candidates, key=lambda i: -self._dense(query_vec, i))[
            : settings.dense_top_k
        ]
        latency.dense_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        bm25 = sorted(candidates, key=lambda i: -self._bm25(terms, i))[: settings.bm25_top_k]
        latency.bm25_ms = (time.perf_counter() - t0) * 1000

        dense_rank = {idx: rank + 1 for rank, idx in enumerate(dense)}
        bm25_rank = {idx: rank + 1 for rank, idx in enumerate(bm25)}

        fused: dict[int, float] = {}
        for ranks in (dense_rank, bm25_rank):
            for idx, rank in ranks.items():
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (settings.rrf_k + rank)

        merged = sorted(fused, key=lambda i: -fused[i])[: settings.rerank_candidates]
        n_candidates = len(merged)

        t0 = time.perf_counter()
        # The reranker is a cross-encoder in the real pipeline. Here: a squashed
        # BM25 score, which at least moves literal matches up the way one would.
        reranked = sorted(merged, key=lambda i: -self._bm25(terms, i))
        scored = [(i, 1 / (1 + math.exp(-(self._bm25(terms, i) - 4) / 2))) for i in reranked]
        latency.rerank_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        kept = [(i, s) for i, s in scored if s >= settings.min_score][: settings.top_k]
        results = [
            RetrievedChunk(
                chunk_id=self.chunks[i].chunk_id,
                doc_id=self.chunks[i].doc_id,
                doc_title=self.chunks[i].doc_title,
                page_start=self.chunks[i].page,
                page_end=self.chunks[i].page,
                section_path=self.chunks[i].section_path,
                text=self.chunks[i].text,
                dense_rank=dense_rank.get(i),
                bm25_rank=bm25_rank.get(i),
                rrf_score=round(fused.get(i, 0.0), 6),
                rerank_score=round(score, 4),
            )
            for i, score in kept
        ]
        latency.pack_ms = (time.perf_counter() - t0) * 1000
        latency.total_ms = (
            latency.embed_ms + latency.dense_ms + latency.bm25_ms
            + latency.rerank_ms + latency.pack_ms
        )
        return results, latency, n_candidates
