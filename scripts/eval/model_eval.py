#!/usr/bin/env python3
"""CPU-only bake-off of candidate embedders/rerankers via llama-server.

Corpus: this repo's own docs (EN fixtures + IT README + plan/architecture as
distractors), one chunk per heading section. Queries: EN + IT, paraphrased,
gold = substring of chunk id or text. Stdlib only.

    python scripts/eval/model_eval.py            # writes scripts/eval/results.md

Reports quality (hit@1, recall@3/@6, MRR) plus cost: wall time, llama-server
CPU-seconds (user+sys) and RSS. Servers run with -ngl 0 on purpose: the
constraint is machines without a GPU.
"""
import json, math, os, re, resource, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = Path(os.environ.get("EVAL_MODEL_DIR", Path.home() / ".ibid/eval-models"))
DEFAULT = Path.home() / ".ibid/models"
THREADS = os.environ.get("EVAL_THREADS", "4")
TOPN = 20
MAXCH = 1500

QWEN_Q = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
EMBEDDERS = {  # name: (file, pooling, query_prefix, doc_prefix)
    "qwen3-emb-0.6B-q8": (DEFAULT / "Qwen3-Embedding-0.6B-Q8_0.gguf", "last", QWEN_Q, ""),
    "embeddinggemma-300M-q4": (MODELS / "embeddinggemma-300M-qat-Q4_0.gguf", None,
                               "task: search result | query: ", "title: none | text: "),
    "bge-m3-q4km": (MODELS / "bge-m3-Q4_K_M.gguf", "cls", "", ""),
}
RERANKERS = {
    "qwen3-rerank-0.6B-q8": DEFAULT / "Qwen3-Reranker-0.6B-Q8_0.gguf",
    "bge-reranker-v2-m3-q4km": MODELS / "bge-reranker-v2-m3-Q4_K_M.gguf",
}

QUERIES = [  # (query, [gold substrings])
    ("why can't I just change the embedding model in an app update", ["embedding-models.md#Why the embedder"]),
    ("how much storage does a million 1024-dimension vectors take", ["embedding-models.md#Choosing a dimension"]),
    ("what does the k constant do in reciprocal rank fusion", ["hybrid-search.md#Reciprocal rank fusion"]),
    ("why keyword search alone misses paraphrased questions", ["hybrid-search.md#Two kinds of miss"]),
    ("make the system say it found nothing instead of returning irrelevant passages", ["reranking.md#The minimum-score gate"]),
    ("can I replace the reranker without rebuilding the index", ["reranking.md#Swapping rerankers"]),
    ("how many candidates to send to the cross-encoder on a laptop without a GPU", ["reranking.md#Candidate count"]),
    ("problem with cutting documents every N tokens", ["chunking-strategies.md#Why fixed-size"]),
    ("send the bigger surrounding passage to the LLM but search on small ones", ["chunking-strategies.md#Parent and child"]),
    ("how many golden questions do I need to start evaluating", ["evaluation-runbook.md#Build the golden set"]),
    ("stop retrieval quality silently regressing in CI", ["evaluation-runbook.md#Regressions are the point"]),
    ("what to do with citations that don't match retrieved passages", ["retrieval-augmented-generation.md#Grounding", "citazioni validate"]),
    ("questions about the main themes across all reports", ["retrieval-augmented-generation.md#When retrieval is the wrong"]),
    ("should language filters run before or after retrieval", ["hybrid-search.md#Filters belong before fusion"]),
    ("which license to avoid for PDF parsing in a distributed product", ["License guard"]),
    ("perché non posso cambiare il modello di embedding senza reindicizzare", ["embedding-models.md#Why the embedder", "Migration rule"]),
    ("quanti candidati passare al reranker su un computer senza GPU", ["reranking.md#Candidate count"]),
    ("cosa succede quando nessun passaggio è abbastanza rilevante", ["reranking.md#The minimum-score gate"]),
    ("dividere i documenti seguendo titoli e struttura invece di token fissi", ["chunking-strategies.md#Structure first"]),
    ("quali metriche usare per valutare il recupero dei passaggi", ["evaluation-runbook.md#Metrics that earn"]),
    ("come avviare l'app in modalità sviluppo", ["README.md#Avvio rapido"]),
    ("come scarico i modelli GGUF per lo sviluppo", ["README.md#Modelli locali"]),
    ("quali modelli locali vengono distribuiti insieme all'app", ["distribuirà localmente"]),
    ("come si scartano le citazioni non valide nelle risposte", ["retrieval-augmented-generation.md#Grounding", "citazioni validate"]),
]


def build_corpus():
    files = sorted((ROOT / "fixtures/docs").glob("*.md")) + [
        ROOT / "README.md", ROOT / "docs/architecture.md", ROOT / "plan/ibid-build-plan.md"]
    out = []
    for f in files:
        title, parts, cur = f.stem, [], None
        for line in f.read_text().splitlines():
            if line.startswith("# ") and not cur:
                title = line[2:].strip()
            elif re.match(r"#{2,3} ", line):
                cur = [line.lstrip("# ").strip(), []]
                parts.append(cur)
            elif cur:
                cur[1].append(line)
        for head, body in parts:
            text = "\n".join(body).strip()
            if len(text) > 60:
                out.append({"id": f"{f.name}#{head}", "text": f"{title} > {head}\n{text}"[:MAXCH]})
    return out


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))


class Server:
    def __init__(self, model, port, extra):
        self.port = port
        self.before = resource.getrusage(resource.RUSAGE_CHILDREN)
        self.p = subprocess.Popen(
            ["llama-server", "-m", str(model), "--port", str(port), "--host", "127.0.0.1",
             "-ngl", "0", "-t", THREADS, "-c", "4096", "-b", "2048", "-ub", "2048", "-np", "1", *extra],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(240):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                return
            except Exception:
                time.sleep(0.5)
        raise RuntimeError(f"{model} did not start")

    def stop(self):
        rss = int(subprocess.run(["ps", "-o", "rss=", "-p", str(self.p.pid)], capture_output=True, text=True).stdout.strip() or 0)
        self.p.terminate(); self.p.wait()
        a = resource.getrusage(resource.RUSAGE_CHILDREN)
        return round(a.ru_utime + a.ru_stime - self.before.ru_utime - self.before.ru_stime, 1), rss // 1024


def norm(v):
    n = math.sqrt(sum(x * x for x in v)) or 1
    return [x / n for x in v]


def embed(port, texts):
    out = []
    for i in range(0, len(texts), 8):
        r = post(port, "/v1/embeddings", {"input": texts[i:i + 8], "model": "m"})
        out += [norm(d["embedding"]) for d in sorted(r["data"], key=lambda d: d["index"])]
    return out


def metrics(rankings, corpus):
    """rankings: per query list of corpus idx, best first."""
    h1 = r3 = r6 = mrr = 0
    for (_, gold), rank in zip(QUERIES, rankings):
        hit = [any(g in corpus[i]["id"] or g in corpus[i]["text"] for g in gold) for i in rank]
        pos = hit.index(True) + 1 if True in hit else None
        h1 += pos == 1; r3 += bool(pos and pos <= 3); r6 += bool(pos and pos <= 6)
        mrr += 1 / pos if pos else 0
    n = len(QUERIES)
    return {"hit@1": h1 / n, "R@3": r3 / n, "R@6": r6 / n, "MRR": mrr / n}


def main():
    corpus = build_corpus()
    print(f"{len(corpus)} chunks, {len(QUERIES)} queries, threads={THREADS}", flush=True)
    rows, cands = [], {}
    for name, (path, pool, qp, dp) in EMBEDDERS.items():
        if not path.exists():
            print("skip", name); continue
        s = Server(path, 8770, ["--embedding", *(["--pooling", pool] if pool else [])])
        t = time.time(); dv = embed(8770, [dp + c["text"] for c in corpus]); t_corpus = time.time() - t
        t = time.time(); qv = embed(8770, [qp + q for q, _ in QUERIES]); t_q = (time.time() - t) / len(QUERIES)
        cpu, rss = s.stop()
        ranks = [sorted(range(len(corpus)), key=lambda i: -sum(a * b for a, b in zip(q, dv[i]))) for q in qv]
        cands[name] = ranks
        rows.append({"stage": f"embed {name}", **metrics(ranks, corpus), "sec": round(t_corpus, 1),
                     "ms/q": round(t_q * 1000), "cpu_s": cpu, "rss_mb": rss})
        print(rows[-1], flush=True)
    for rn, rpath in RERANKERS.items():
        if not rpath.exists():
            print("skip", rn); continue
        s = Server(rpath, 8771, ["--reranking"])
        for en, ranks in cands.items():
            t = time.time(); out = []
            for (q, _), rank in zip(QUERIES, ranks):
                top = rank[:TOPN]
                r = post(8771, "/v1/rerank", {"query": q, "documents": [corpus[i]["text"] for i in top]})
                sc = {x["index"]: x["relevance_score"] for x in r["results"]}
                out.append([top[j] for j in sorted(sc, key=lambda j: -sc[j])] + rank[TOPN:])
            el = time.time() - t
            rows.append({"stage": f"{en} + {rn}", **metrics(out, corpus), "sec": round(el, 1),
                         "ms/q": round(el * 1000 / len(QUERIES)), "cpu_s": None, "rss_mb": None})
            print(rows[-1], flush=True)
        cpu, rss = s.stop()
        rows.append({"stage": f"[{rn} total server]", "sec": None, "cpu_s": cpu, "rss_mb": rss})
    keys = ["stage", "hit@1", "R@3", "R@6", "MRR", "sec", "ms/q", "cpu_s", "rss_mb"]
    fmt = lambda v: "" if v is None else (f"{v:.2f}" if isinstance(v, float) and v <= 1 else str(v))
    md = "| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n" + "\n".join(
        "| " + " | ".join(fmt(r.get(k)) for k in keys) + " |" for r in rows)
    (ROOT / "scripts/eval/results.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    sys.exit(main())
