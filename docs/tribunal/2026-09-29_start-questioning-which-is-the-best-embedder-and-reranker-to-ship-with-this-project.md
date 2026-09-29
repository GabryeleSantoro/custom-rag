**Invocation:** /tribunal:deliberate Start questioning which is the best embedder and reranker to ship with this project

## ⚖️ Tribunal Verdict

**Topic:** Best embedder and reranker to ship with custom-rag (framed as: keep the planned Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B GGUF pair, served by two `llama-server` processes?)
**Domain:** technical (ML retrieval / product distribution)
**Session ID:** 3b51a4dd-f3e2-4fe5-afc6-bea14a60c1e9
**Depth:** full
**Multi-agent:** no

**Grounding read:** `plan/ibid-build-plan.md`, `scripts/fetch-models.sh`, `scripts/dev/serve-models.sh`, `docs/superpowers/notes/2026-09-17-s0-reranker-gate.md`, README. Facts: plan ships Qwen3-Embedding-0.6B (Q8_0, dim 1024) + Qwen3-Reranker-0.6B (ggml-org conversion), 4B reranker as optional download; reranker gate passed only after swapping to a reranker-aware GGUF (first run Spearman -0.33); measured machine ran CPU-only (no Metal); embedder swap forces full re-index, reranker swap does not. Not in repo: any head-to-head retrieval eval against alternatives, any Italian/multilingual eval (README is Italian). Alternatives named below (bge-m3, bge-reranker-v2-m3, EmbeddingGemma, jina-v3) come from prior knowledge, **not verified this session**.

### Final Verdict
**Decision:** Lean support: keep Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B as the shipped default, with Reranker-4B optional. Condition: before the embedder is frozen, run one small retrieval eval on the project's own fixtures (include Italian queries/docs) against at least one challenger embedder (bge-m3 is the obvious one) and one challenger reranker (bge-reranker-v2-m3). Reranker choice is cheap to revisit; embedder choice is not, so spend the eval budget there. Do not add a model-selection UI or ONNX fallback until the eval shows a need.
**Confidence:** 33%
**Dissent:** Devil's Advocate (toward reject): the pick was made on convenience (one vendor, one runtime, one GGUF path), not on measured retrieval quality; 0.6B may underperform on Italian and long technical docs, and the reranker GGUF already proved fragile once. Logician (neutral): the S0 gate proves parity with the HF reference implementation, not that this pair is *best*; "best" is currently unsupported by evidence in the repo.

### Votes
| Persona | Position | Lean | Conf. | v_i | Rationale |
|---|---|---|---|---|---|
| Domain Expert | support | n/a | 72% | +1 | Qwen3 pair is Apache-2.0, multilingual, 1024-d, runs via one runtime with CPU fallback, and the reranker path is already validated against the reference. |
| Devil's Advocate | conditional | toward_reject | 60% | -0.5 | No eval on real/Italian data; GGUF reranker nearly shipped garbage scores once; 0.6B rerank on CPU-only profile is a latency and quality gamble. |
| Systems Thinker | conditional | toward_support | 65% | +0.5 | Embedder is a one-way door (index_meta migration, re-index), reranker is two-way; ship Qwen3 but keep model id/dim in index_meta and eval before v1 freeze. |
| Logician | conditional | neutral | 55% | 0 | "Best" is a comparative claim with no comparison in evidence; passing a parity gate does not entail superiority. Conclusion currently rests on priors. |
| Mediator | conditional | toward_support | 60% | +0.5 | Both camps agree on the default; they disagree only on whether to measure first. A cheap eval satisfies both. |

### Reasoning Trail
- v_i: +1, -0.5, +0.5, 0, +0.5 ; w_i: 0.72, 0.60, 0.65, 0.55, 0.60 (Σw = 3.12)
- Σ(v_i·w_i) = 0.72 - 0.30 + 0.325 + 0 + 0.30 = 1.045 → v̄ = 0.335
- Weighted consensus strength = 33% (> +0.15 → lean support; < 80 so Dissent included)
- Minority (negative) weight 0.60 / 3.12 = 19% (< 35% threshold, dissent kept anyway because strength < 80)
- Decision-follows-math audit: support-leaning text, v̄ > 0.15 ✔
- Weak spots: confidence is capped by missing comparative evidence, not by disagreement about the default.

---
*The full deliberation was produced internally. Use `/tribunal:deliberate --full-log …` (or `--verbose`, or ask for a full log) on a later run to include topic analysis, panel, arguments, cross-examination, and deliberation **in the saved file**.*
