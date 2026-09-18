# S0 — reranker gate result

**Verdict: FAIL.** `llama-server`'s `/v1/rerank` endpoint does not rank the
50-pair fixture corpus the way the reference HuggingFace implementation of
`Qwen/Qwen3-Reranker-0.6B` does. Per the task brief: STOP. Do not start Task 4.
The `models/` design needs to change (the brief's stated fallback is ONNX
Runtime for the reranker) and the spec needs to be revisited.

## Models used

- Embedder: `Qwen3-Embedding-0.6B-Q8_0.gguf`, fetched from
  `https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf`
  (as specified in the brief; this URL resolved fine).
  SHA-256: `06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439`.

- Reranker: `Qwen3-Reranker-0.6B-Q8_0.gguf` — **deviation from the brief.**
  The brief's URL, `https://huggingface.co/Qwen/Qwen3-Reranker-0.6B-GGUF/resolve/main/...`,
  does not 404, it returns **HTTP 401** ("Invalid username or password") on
  both the API and the web page, with no auth configured on this machine.
  Checked via the HF search API: no repo named
  `Qwen/Qwen3-Reranker-0.6B-GGUF` exists under the `Qwen` org (only
  `Qwen/Qwen3-Reranker-0.6B`, the original safetensors checkpoint, exists
  there — Qwen never published an official GGUF conversion of the reranker,
  unlike the embedder). This is a 401-on-nonexistent-repo, not a filename
  mismatch inside an existing repo, so it falls outside the brief's literal
  "if a URL 404s, list the repo's files and use the actual Q8_0 filename"
  guidance — there is no repo to list files from.

  Used instead: `mradermacher/Qwen3-Reranker-0.6B-GGUF`, file
  `Qwen3-Reranker-0.6B.Q8_0.gguf`
  (`https://huggingface.co/mradermacher/Qwen3-Reranker-0.6B-GGUF/resolve/main/Qwen3-Reranker-0.6B.Q8_0.gguf`),
  saved locally as `Qwen3-Reranker-0.6B-Q8_0.gguf` to match `serve-models.sh`.
  This repo's README documents it as a "static quant" (plain
  `convert_hf_to_gguf.py` conversion, no imatrix, no architecture change) of
  `https://huggingface.co/Qwen/Qwen3-Reranker-0.6B` — the same base model the
  spike's `MODEL_ID` references — at Q8_0. No other quantisation was
  substituted. Its Q8_0 file size (639,150,752 bytes) is consistent with a
  faithful conversion of the same 0.6B parameter count as the embedder's
  official Q8_0 (639,150,592 bytes). SHA-256:
  `c525a7449243f690a7062e6377d6cf5adbb289354bd4316312367cd20e187ab7`.

  This substitution is a plausible confound for the FAIL verdict below and is
  called out explicitly in the task report; it was not "fixed" or re-tried
  with a different source, per the brief's instruction not to chase the
  number once the gate fails.

## `llama-server --version`

```
version: 0.1.2-dev (build 10520, commit cd644c395)
built with AppleClang 16.0.0.16000026 for Darwin x86_64
```

## Machine and GPU backend

- Machine: MacBook Pro, Apple M1 Max (`arm64`), Darwin 25.6.0 (macOS).
- GPU backend: **none / CPU only** — not Metal. `llama-server`'s own startup
  log states this explicitly for both server instances:

  ```
  warning: no usable GPU found, --gpu-layers option will be ignored
  warning: one possible reason is that llama.cpp was compiled without GPU support
  warning: consult docs/build.md for compilation instructions
  ```

  This is consistent with the `--version` output above reporting the binary
  as built `for Darwin x86_64` on an `arm64` host: the installed
  `/usr/local/bin/llama-server` is an x86_64 build (running under Rosetta),
  compiled without Metal/GPU support, so `-ngl 999` in `serve-models.sh` was
  silently ignored and both models ran on CPU.

## Step 4 — embedder check

```
$ curl -s http://127.0.0.1:8770/v1/embeddings \
    -H 'Content-Type: application/json' \
    -d '{"input":["reranking reorders candidates"],"model":"qwen3-embedding"}' \
  | python3 -c 'import json,sys,math; v=json.load(sys.stdin)["data"][0]["embedding"]; print("dim", len(v), "norm", round(math.sqrt(sum(x*x for x in v)),4))'

dim 1024 norm 1.0
```

`dim` matches `EMBED_DIM = 1024`. The norm is ~1.0, i.e. `llama-server`
already returns L2-normalized vectors for this embedder — Task 4's
`models/embed.py` does **not** need to L2-normalize client-side for this
model/server combination.

## Step 5 — gate result

Command:

```
./scripts/fetch-models.sh
./scripts/dev/serve-models.sh &
python3 -m venv /tmp/s0 && /tmp/s0/bin/pip install torch transformers httpx scipy
/tmp/s0/bin/python scripts/spikes/rerank_check.py
```

Output (verbatim):

```json
{
  "spearman": -0.3305,
  "pairwise_agreement": 0.3894,
  "top1_match": 0,
  "pass": false
}
```

`spearman` is not just below the 0.9 threshold, it is negative — the
llama.cpp reranker ranking is anti-correlated with the reference HF
implementation's yes/no-softmax ranking over this 50-pair fixture corpus, not
merely noisier than it. `top1_match` is 0: the two implementations do not
even agree on which single passage is the best match for any of the queries
in aggregate ranking.

## Conclusion

Gate fails (`pass: false`, `spearman < 0.9`, `pairwise_agreement < 0.9`).
Per the task brief, Task 4 and onward are blocked. The `models/` design as
specified (drive the reranker entirely through `llama-server`'s built-in
`--reranking` support) does not reproduce the reference scoring behavior in
this environment, whether that is because of the community GGUF conversion
used, this build of `llama-server`, or the underlying llama.cpp
implementation of Qwen3-Reranker's yes/no-logit reranking task. That
diagnosis is out of scope for this gate: per the brief, the next step is for
the plan owner to decide, not for this task to guess and retry prompt
formats, thresholds or scoring math.
