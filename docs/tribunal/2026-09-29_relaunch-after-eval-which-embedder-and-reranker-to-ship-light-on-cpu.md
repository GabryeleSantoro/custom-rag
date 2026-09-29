**Invocation:** /tribunal:deliberate Relaunch after eval: which embedder and reranker to ship, must be light on CPU and small (user: "Must be light on CPU so keep it small. Also make yourself an eval based on your needs. then relaunch the tribunal to decide what to do")

## ⚖️ Tribunal Verdict

**Topic:** Ship EmbeddingGemma-300M (Q4) as default embedder with the reranker off by default / opt-in, instead of the planned Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B pair, under a "light on CPU, small" constraint
**Domain:** technical (local ML retrieval, distribution)
**Session ID:** 922b8571-07c4-4246-b54c-a1d9cb22739c
**Depth:** full
**Multi-agent:** no

**Evidence (own eval, `scripts/eval/model_eval.py`, full table in `scripts/eval/results.md`):** 59 heading-level chunks (EN fixtures + IT README + plan/architecture as distractors), 24 paraphrased EN/IT queries with gold sections, llama-server CPU-only (`-ngl 0`, 4 threads, Apple M1 Max).

| Stage | hit@1 | R@3 | R@6 | MRR | ms/query | embed CPU-s | RSS MB |
|---|---|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B Q8 | 0.92 | 1.00 | 1.00 | 0.96 | 52 | 131 | 2687 |
| EmbeddingGemma-300M Q4 | 0.96 | 1.00 | 1.00 | 0.98 | 12 | 52 | 1201 |
| bge-m3 Q4_K_M | 0.71 | 0.83 | 0.92 | 0.80 | 33 | 113 | 1314 |
| any embedder + Qwen3-Reranker-0.6B (top 20) | 0.92 | 0.96 | 1.00 | 0.95 | 15,000–18,000 | server 4426 | 2638 |
| any embedder + bge-reranker-v2-m3 Q4 (top 20) | 0.92 | 1.00 | 1.00 | 0.95 | 8,900–10,200 | server 2659 | 1329 |

Reading: reranking added nothing on this set (it slightly lowered MRR for the best embedder) and cost 9-18 s per query on CPU for 20 candidates. Caveats: tiny corpus with ceiling effect (one query = 4 points, so 0.92 vs 0.96 is one query); the rerank latency may be inflated by my server flags (-np 1, -ub 2048, ~1500-char docs) and was not tuned; bge-m3's weak score may be the Q4 quant or my pooling setup, not investigated. Not verified this session: EmbeddingGemma's license terms for redistribution and its context limit (believed Gemma terms of use, ~2K tokens) and output dim (believed 768, not the 1024 in `EMBED_DIM`).

### Final Verdict
**Decision:** Lean support. Make EmbeddingGemma-300M Q4 the default embedder (about 2.5x less CPU and 2x less RAM than Qwen3-0.6B, and not worse on this eval). Ship with the reranker **off by default**, as an opt-in download; when enabled, use few candidates (about 8) and re-measure. Drop bge-m3 and bge-reranker from consideration. Before freezing: (1) confirm Gemma's license allows redistribution, or download it on first run instead of bundling it; (2) set `EMBED_DIM` to its real dimension now, while no user index exists yet; (3) replace the reranker-based "not found" gate with a calibrated cosine-score threshold; (4) extend the eval with a larger, harder set, ideally private Italian documents, before v1. Keep Qwen3-Embedding-0.6B as the fallback if the license or the harder eval says no.
**Confidence:** 32%
**Dissent:** Devil's Advocate (toward reject): 24 queries on 59 chunks cannot separate the models; the reranker's zero gain is a ceiling artifact, and dropping it removes the README's "not found" gate; a Gemma license restriction would force a change after users have indexes. Logician (neutral): the data supports "not worse and cheaper", not "better", and does not show rerankers are useless.

### Votes
| Persona | Position | Lean | Conf. | v_i | Rationale |
|---|---|---|---|---|---|
| Domain Expert | support | n/a | 62% | +1 | Best hit@1/MRR at a fraction of the CPU and RAM; reranker cost of 9-18 s/query is incompatible with a light-CPU product for zero measured gain. |
| Devil's Advocate | conditional | toward_reject | 58% | -0.5 | Tiny set hides differences; Gemma license and context limit unverified; no reranker means no "not found" gate; bge-m3 result may be my setup error. |
| Systems Thinker | conditional | toward_support | 60% | +0.5 | Embedder is the one-way door, so choose the small one now while no index exists; smaller embedder also makes re-index cheap; reranker stays a reversible opt-in. |
| Logician | conditional | neutral | 55% | 0 | "Cheaper and no worse" is valid under the stated constraint; "better" and "reranker useless" do not follow from a ceiling-limited sample. |
| Mediator | conditional | toward_support | 60% | +0.5 | Small default satisfies the user's constraint; opt-in reranker and a harder eval before v1 keep the sceptics' concerns addressed. |

### Reasoning Trail
- v_i: +1, -0.5, +0.5, 0, +0.5 ; w_i: 0.62, 0.58, 0.60, 0.55, 0.60 (Σw = 2.95)
- Σ(v_i·w_i) = 0.62 - 0.29 + 0.30 + 0 + 0.30 = 0.93 → v̄ = 0.315
- Weighted consensus strength = 32% (> +0.15 → lean support; < 80 so Dissent included)
- Minority (negative) weight 0.58 / 2.95 = 20% (< 35%)
- Decision-follows-math audit: support-leaning text, v̄ > 0.15 ✔
- Change from previous run (33%, keep Qwen3 pair): evidence now favors the smaller embedder; confidence stays low because the eval is small and two facts are unverified.

---
*The full deliberation was produced internally. Use `/tribunal:deliberate --full-log …` (or `--verbose`, or ask for a full log) on a later run to include topic analysis, panel, arguments, cross-examination, and deliberation **in the saved file**.*
