# S0 — reranker gate result

**Verdict: PASS** (after a diagnostic round — see below). `llama-server`'s
`/v1/rerank` endpoint, given a reranker-aware GGUF conversion of
`Qwen/Qwen3-Reranker-0.6B`, reproduces the reference HuggingFace
implementation's ranking on the 50-pair fixture corpus well above the 0.9
bar. Task 4 onward may proceed.

The first run of this gate (same day) FAILED (`spearman: -0.3305`). A bounded
diagnostic round (below) established that the failure was caused entirely by
the specific GGUF file used, not by `llama-server`'s reranking support or by
the fixture corpus. Both runs are recorded here in full, per the diagnostic
round's own instruction not to erase the failing evidence.

## Models used (final, working configuration)

- Embedder: `Qwen3-Embedding-0.6B-Q8_0.gguf`, from
  `https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf`
  (as specified in the brief; unchanged across both runs).
  SHA-256: `06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439`.

- Reranker: `Qwen3-Reranker-0.6B-Q8_0.gguf`, fetched from
  `https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/resolve/main/qwen3-reranker-0.6b-q8_0.gguf`
  and saved locally under that name to match `serve-models.sh`'s convention.
  SHA-256: `22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48`.
  This is `ggml-org`'s (the llama.cpp project's own HuggingFace account)
  conversion via their `GGUF-my-repo` tool, which — unlike a plain
  `convert_hf_to_gguf.py` static quant — extracts Qwen3-Reranker's
  `cls.output.weight` classifier tensor and rank-pooling metadata (see Root
  cause below). `fetch-models.sh` and `scripts/dev/serve-models.sh` are
  unchanged in structure; only the reranker URL in `fetch-models.sh` was
  updated to this repo.

  The brief's own URL, `Qwen/Qwen3-Reranker-0.6B-GGUF`, still does not
  exist (confirmed via the HF search API: no such repo under the `Qwen` org,
  only `Qwen/Qwen3-Reranker-0.6B`, the original safetensors checkpoint). That
  finding is unchanged from the first run.

## `llama-server --version`

```
version: 0.1.2-dev (build 10520, commit cd644c395)
built with AppleClang 16.0.0.16000026 for Darwin x86_64
```

## Machine and GPU backend

- Machine: MacBook Pro, Apple M1 Max (`arm64`), Darwin 25.6.0 (macOS).
- GPU backend: **none / CPU only** — not Metal. `llama-server`'s own startup
  log states this explicitly for both server instances, in every run:

  ```
  warning: no usable GPU found, --gpu-layers option will be ignored
  warning: one possible reason is that llama.cpp was compiled without GPU support
  warning: consult docs/build.md for compilation instructions
  ```

  The installed `/usr/local/bin/llama-server` is an x86_64 build (running
  under Rosetta on this arm64 host), compiled without Metal/GPU support, so
  `-ngl 999` in `serve-models.sh` is silently ignored and both models run on
  CPU. This is unrelated to the reranker gate's pass/fail outcome (verified:
  the correct conversion passes on this same CPU-only setup) but is relevant
  to later tasks' performance assumptions.

## Step 4 — embedder check

```
$ curl -s http://127.0.0.1:8770/v1/embeddings \
    -H 'Content-Type: application/json' \
    -d '{"input":["reranking reorders candidates"],"model":"qwen3-embedding"}' \
  | python3 -c 'import json,sys,math; v=json.load(sys.stdin)["data"][0]["embedding"]; print("dim", len(v), "norm", round(math.sqrt(sum(x*x for x in v)),4))'

dim 1024 norm 1.0
```

`dim` matches `EMBED_DIM = 1024`. Norm ~1.0: `llama-server` already returns
L2-normalized vectors for this model, so Task 4's `models/embed.py` does not
need to L2-normalize client-side. Reproduced identically before and after
the reranker fix (this endpoint/model was never in question).

## Run 1 (original): FAIL, using a plain static-quant reranker GGUF

Model used: `mradermacher/Qwen3-Reranker-0.6B-GGUF`, file
`Qwen3-Reranker-0.6B.Q8_0.gguf` — a "static quant" (the repo's own README's
words: plain `convert_hf_to_gguf.py` conversion, no imatrix, no architecture
change) of `Qwen/Qwen3-Reranker-0.6B`. SHA-256:
`c525a7449243f690a7062e6377d6cf5adbb289354bd4316312367cd20e187ab7`.

```json
{
  "spearman": -0.3305,
  "pairwise_agreement": 0.3894,
  "top1_match": 0,
  "pass": false
}
```

`llama-server`'s startup log for this model included:

```
0.01.170.368 W llama_init_from_model: model default pooling_type is [-1], but [4] was specified
```

i.e. the GGUF itself carries no pooling-type metadata (`-1`); `--reranking`
force-overrides it to `4` (`LLAMA_POOLING_TYPE_RANK`) at the server level.
This warning **did not appear** in Run 2 below, using the correct
conversion — consistent with the correct conversion shipping proper
pooling/classifier metadata baked in.

## Diagnostic round

Ordered cheapest-first, per instruction, and not stopped at the first
conclusive-looking result.

### 1. Are the scores degenerate?

Dumped the raw 50 `relevance_score` values from Run 1's (broken) llama
server and the 50 reference `P(yes)` values, computed via the same
`pairs()`/`llama_scores()`/`reference_scores()` functions in the committed
`scripts/spikes/rerank_check.py` (imported unchanged, no scoring math
touched):

```
llama (relevance_score): min=1.16163e-13 max=3.03888e-06 mean=1.3401e-07 stdev=4.58844e-07
reference (softmax P(yes)): min=6.19888e-05 max=0.996094 mean=0.0562434 stdev=0.192668
llama raw range observed: [1.16163e-13, 3.03888e-06]
```

First 10 of 50 pairs, side by side (query truncated to 40 chars, passage to
50):

```
[0] llama=5.41612e-10  reference=0.00212097  'how does reranking improve retrieval pre' | 'Splitting every N tokens is simple and ignores the'...
[1] llama=2.64633e-08  reference=0.000572205  'how does reranking improve retrieval pre' | "A better order is to split on the document's own s"...
[2] llama=1.27854e-11  reference=0.019165  'how does reranking improve retrieval pre' | 'Small passages retrieve well because their vector '...
[3] llama=3.53317e-09  reference=6.38962e-05  'how does reranking improve retrieval pre' | 'Deduplicating parents matters here. Three winning '...
[4] llama=1.16163e-13  reference=0.00805664  'how does reranking improve retrieval pre' | 'An embedder maps text to a fixed-length vector so '...
[5] llama=1.00413e-12  reference=0.00113678  'how does reranking improve retrieval pre' | 'Larger vectors carry more information and cost mor'...
[6] llama=3.82634e-13  reference=0.00180817  'how does reranking improve retrieval pre' | 'Vectors from two different embedders are not compa'...
[7] llama=1.19138e-07  reference=0.00288391  'how does reranking improve retrieval pre' | 'That check is cheap and it prevents the worst clas'...
[8] llama=7.00607e-09  reference=0.000431061  'how does reranking improve retrieval pre' | 'Fifty questions with known answers is enough to st'...
[9] llama=1.16146e-07  reference=0.000295639  'how does reranking improve retrieval pre' | 'Questions should come from the corpus, not from im'...
```

Answer: llama's raw scores are a bounded `[0,1]` probability (it is a
softmax-derived `relevance_score`, not an unbounded logit) but are
functionally degenerate: every one of the 50 values is below `3.1e-06`,
spanning roughly seven orders of magnitude entirely within "near zero,"
uncorrelated with the reference's genuine dynamic range (`6.2e-05` to
`0.996`, spanning relevant and irrelevant pairs sensibly). Not a
zero-variance constant, but functionally noise clustered at the floor of the
range — the signature of a classifier head that isn't actually there.

### 2. Does this build of llama.cpp actually support Qwen3-Reranker?

Yes, but only with a conversion that carries the right tensor and metadata.
Qwen3-Reranker is a causal LM scored by the softmax of its "yes"/"no" logits
at the last token, not a BERT-style cross-encoder with a native
classification head. llama.cpp's `--reranking` path uses
`LLAMA_POOLING_TYPE_RANK`, which expects a `cls.output.weight`
`[hidden_dim, 2]` tensor projecting the final hidden state to
`[P(no), P(yes)]`-shaped logits. The official `convert_hf_to_gguf.py`
detects the Qwen3-Reranker architecture and extracts this tensor (and the
rank-pooling metadata) from the base model's `lm_head`; a plain/generic
static-quant conversion run through an older or reranker-unaware conversion
path does not, leaving `--reranking` to force RANK pooling onto a model with
no valid classifier weights for it — hence the near-zero garbage.

This is a documented, known llama.cpp issue, not a guess:

- [ggml-org/llama.cpp#16407 — "server/rerank output result is wrong with
  most models include qwen3-Rerank"](https://github.com/ggml-org/llama.cpp/issues/16407):
  reports the exact same near-zero-garbage-score signature (values like
  `1.1353239058953662E-28`, `3.111864641067425E-29`) on Qwen3-Reranker and
  other rerankers.
- [VooDisss's gist, "llama-server models.ini guide for Qwen3 reranker +
  embedding + chat models. Fix for Qwen3-Reranker GGUF producing near-zero
  scores (4.5e-23) with
  llama.cpp"](https://gist.github.com/VooDisss/42bce4eb5c76d3c325633886c5e348ee):
  documents this exact failure mode and its cause — "missing `cls.output.weight`
  classifier tensor, `pooling_type=RANK` metadata" in most community
  conversions — and the fix (official `convert_hf_to_gguf.py`, which
  "automatically detects Qwen3-Reranker and extracts the classifier tensor +
  metadata").
- [ggml-org/llama.cpp#28876 — "server: allow RANK pooling batch splitting for
  causal LLM rerankers (ie. Qwen3 and
  Qwen3-VL)"](https://github.com/ggml-org/llama.cpp/pull/28876): recent,
  active upstream work on exactly this model family's reranking path,
  confirming it is a live area of the codebase, not long-abandoned.
- [Qwen/Qwen3-Reranker-0.6B discussion #22, "Working GGUF for llama.cpp
  (native Windows/Linux, no WSL
  needed)"](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/discussions/22):
  community discussion converging on the same root cause and recommending
  conversions that include the classifier tensor.

`llama-server`'s own startup log in Run 1 (`model default pooling_type is
[-1], but [4] was specified`) is consistent with this: the mradermacher
static quant carries no pooling-type metadata at all, so `--reranking` is
overriding a GGUF that was never built to be used this way.

### 3. Is there a conversion documented as working with llama.cpp reranking?

Yes: `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF` — the llama.cpp project's own
HuggingFace account, converted via their `GGUF-my-repo` tool, tagged
`text-ranking`. (`Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp` is a second,
independently-documented working conversion found in the same search, not
used here since the ggml-org one is more authoritative — same upstream
project — and was the one Qwen's own community discussion pointed to for a
"working GGUF for llama.cpp".)

Fetched into a scratch model directory (`RAGCORE_MODEL_DIR=/tmp/s0-models-fixed`,
not committed) and pointed `serve-models.sh` at it via the existing env-var
convention, without touching the committed scripts. `llama-server`'s startup
log for this file **did not** show the "model default pooling_type is [-1]"
override warning seen in Run 1 — consistent with this conversion shipping
correct pooling metadata.

Single-pair sanity check immediately showed a dramatic difference from Run 1:

```
$ curl -s http://127.0.0.1:8771/v1/rerank -H 'Content-Type: application/json' -d '{
  "query":"what chunk size works best for embeddings",
  "documents":["Small passages retrieve well ... embed small children of roughly 256 tokens ... parent passage of roughly 1024 tokens ...",
               "Jupiter is the largest planet in the solar system and has at least ninety-five known moons, the largest being Ganymede."],
  "top_n":2}'

{"results":[{"index":0,"relevance_score":0.9835249781608582},{"index":1,"relevance_score":5.922311174799688e-05}]}
```

Then re-ran the **unmodified** `scripts/spikes/rerank_check.py` gate against
this server:

```json
{
  "spearman": 0.9984,
  "pairwise_agreement": 0.9861,
  "top1_match": 1,
  "pass": true
}
```

This was then reproduced against the actual committed model path (see
"Run 2 (final)" below) after replacing the broken file in
`~/.custom-rag/models` and updating `fetch-models.sh`'s reranker URL — same
result, byte-identical model (verified by SHA-256).

### 4. Sanity-check both sides against ground truth

Six hand-made pairs — 3 obviously relevant (query and passage on the exact
same topic, drawn from the fixture docs), 3 obviously irrelevant (a RAG
query against a passage about bread, the Amazon rainforest, or Jupiter) —
scored by both the reference implementation and llama, first against the
broken Run 1 server, then against the fixed server:

| label | expected | reference P(yes) | llama (Run 1, broken) | llama (Run 2, fixed) |
|---|---|---|---|---|
| R1-relevant | relevant | 0.000968933 | 2.33013e-07 | 0.00117349 |
| R2-relevant | relevant | 0.972656 | 1.26714e-10 | 0.983202 |
| R3-relevant | relevant | 0.00382996 | 3.88085e-08 | 0.00409331 |
| I1-irrelevant | irrelevant | 2.22921e-05 | 1.69759e-08 | 2.23507e-05 |
| I2-irrelevant | irrelevant | 4.26769e-05 | 4.94243e-13 | 4.46831e-05 |
| I3-irrelevant | irrelevant | 5.22137e-05 | 2.18692e-09 | 5.31919e-05 |

Two things this table establishes on its own, independent of the 50-pair
fixture corpus:

- **The broken (Run 1) llama ranks its single clearest positive example
  (R2, reference confidence 97%, an explicit "256/1024 tokens" answer to
  "what chunk size works best for embeddings") second-to-last of all six
  pairs** — below two of the three deliberately irrelevant passages (I1,
  I3). A reranker that inverts its most unambiguous positive signal below
  unrelated passages about bread and Jupiter is not "noisier" than the
  reference, it is not implementing relevance scoring at all on this
  conversion. This rules out "the fixture corpus is too homogeneous to rank"
  as the explanation for Run 1's failure — these six pairs are not
  homogeneous, and llama still inverts them.
- **The fixed (Run 2) llama reproduces the reference almost exactly** on all
  six pairs, including matching the reference's own within-"relevant"
  ordering (R2 ≫ R3 > R1, matching reference's R2 ≫ R3 > R1) and getting
  within ~5% relative on every irrelevant pair. This is not merely "ranks
  the same," the absolute probabilities are nearly numerically identical to
  the reference's own softmax outputs.

## Run 2 (final): PASS, using the ggml-org conversion, at the committed model path

After updating `scripts/fetch-models.sh`'s reranker URL to
`ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF` and re-running it against the real
`~/.custom-rag/models` (`RAGCORE_MODEL_DIR` unset, i.e. the default path
future runs will use), then `scripts/dev/serve-models.sh` unmodified:

```
$ ./scripts/fetch-models.sh
have Qwen3-Embedding-0.6B-Q8_0.gguf
fetching Qwen3-Reranker-0.6B-Q8_0.gguf
[...]
models in /Users/gabrielesantoro/.custom-rag/models
-rw-r--r-- 1 gabrielesantoro staff 610M Qwen3-Embedding-0.6B-Q8_0.gguf
-rw-r--r-- 1 gabrielesantoro staff 610M Qwen3-Reranker-0.6B-Q8_0.gguf

$ ./scripts/dev/serve-models.sh &
embed pid 58508 :8770 | rerank pid 58509 :8771
# no "model default pooling_type is [-1]" warning this time

$ /tmp/s0/bin/python scripts/spikes/rerank_check.py
{
  "spearman": 0.9984,
  "pairwise_agreement": 0.9861,
  "top1_match": 1,
  "pass": true
}
```

`spearman >= 0.9` and `pairwise_agreement >= 0.9`: **gate passes.** Task 4
onward may proceed using `llama-server`'s `--reranking` support as designed,
provided the reranker model is fetched from a reranker-aware conversion
(now `fetch-models.sh`'s default) rather than an arbitrary community GGUF
repo.

## Conclusion

The instrument (llama.cpp's `--reranking` support, correctly configured
against a conversion that carries the `cls.output.weight` classifier tensor)
is sound and reproduces the reference implementation closely. The original
FAIL was entirely a data-source defect: the brief's specified
`Qwen/Qwen3-Reranker-0.6B-GGUF` repository does not exist, and the community
substitute used in its place (`mradermacher`'s plain static quant) lacks the
tensor and metadata llama.cpp's RANK pooling requires for this model family,
which is a known, documented, currently-being-worked-on area of llama.cpp
(see links above) rather than a fundamental incompatibility. No `models/`
redesign or ONNX Runtime fallback is warranted on this evidence.
