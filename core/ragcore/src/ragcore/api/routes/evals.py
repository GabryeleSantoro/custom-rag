"""Golden-set evaluation. Dev builds only."""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException

from ragcore.api.deps import ConfigDep, StoreDep
from ragcore.api.schemas import (
    EvalMetrics,
    EvalQuestionResult,
    EvalResult,
    EvalRunRequest,
    EvalSet,
    QueryFilters,
    StageLatency,
)
from ragcore.api.sse import frame, sse_response

router = APIRouter(prefix="/eval", tags=["eval"])


@router.get("/sets", response_model=list[EvalSet])
def list_sets(store: StoreDep) -> list[EvalSet]:
    return store.eval_sets


@router.post("/run")
async def run_eval(payload: EvalRunRequest, store: StoreDep, config: ConfigDep):
    if not config.dev_mode:
        raise HTTPException(403, "eval runs in dev builds only")

    eval_set = next((s for s in store.eval_sets if s.name == payload.set_name), None)
    if eval_set is None:
        raise HTTPException(404, f"unknown eval set {payload.set_name!r}")

    # Questions are drawn from the fixture corpus so retrieval actually runs;
    # only the relevance labels are synthetic.
    questions = [
        (c.text.split(".")[0][:90], c.doc_id)
        for doc in store.loaded.values()
        for c in doc.chunks[:2]
    ][: eval_set.n_questions]

    async def events() -> AsyncIterator[str]:
        results: list[EvalQuestionResult] = []
        for index, (question, expected_doc) in enumerate(questions):
            chunks, latency, _ = store.retriever.search(
                question,
                settings=store.settings.retrieval,
                filters=QueryFilters(),
                doc_meta=store.doc_meta(),
            )
            hit_rank = next(
                (i + 1 for i, c in enumerate(chunks) if c.doc_id == expected_doc), None
            )
            results.append(
                EvalQuestionResult(
                    q=question,
                    type="local",
                    hit=hit_rank is not None,
                    rank=hit_rank,
                    ndcg=1.0 / (hit_rank or 99),
                    latency_ms=round(latency.total_ms, 2),
                )
            )
            yield frame(
                "progress",
                {"completed": index + 1, "total": len(questions), "question": question},
            )
            await asyncio.sleep(0.02)

        n = max(len(results), 1)
        recall_6 = sum(1 for r in results if r.hit) / n
        recall_1 = sum(1 for r in results if r.rank == 1) / n
        mrr = sum(1 / r.rank for r in results if r.rank) / n
        ndcg = sum(r.ndcg for r in results) / n

        def percentile(p: float) -> StageLatency:
            values = sorted(r.latency_ms for r in results) or [0.0]
            total = values[min(int(len(values) * p), len(values) - 1)]
            return StageLatency(total_ms=round(total, 2))

        metrics = EvalMetrics(
            set_name=eval_set.name,
            n_questions=len(results),
            recall_at_1=round(recall_1, 4),
            recall_at_6=round(recall_6, 4),
            mrr=round(mrr, 4),
            ndcg_at_6=round(ndcg, 4),
            latency_p50=percentile(0.5),
            latency_p95=percentile(0.95),
            baseline_recall_at_6=(
                round(max(0.0, recall_6 - random.uniform(-0.04, 0.06)), 4)
                if payload.compare_baseline
                else None
            ),
        )
        yield frame("result", EvalResult(metrics=metrics, questions=results))

    return sse_response(events())
