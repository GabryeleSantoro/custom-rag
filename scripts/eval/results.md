| stage | hit@1 | R@3 | R@6 | MRR | sec | ms/q | cpu_s | rss_mb |
|---|---|---|---|---|---|---|---|---|
| embed qwen3-emb-0.6B-q8 | 0.92 | 1.00 | 1.00 | 0.96 | 30.8 | 52 | 131.1 | 2687 |
| embed embeddinggemma-300M-q4 | 0.96 | 1.00 | 1.00 | 0.98 | 12.5 | 12 | 51.8 | 1201 |
| embed bge-m3-q4km | 0.71 | 0.83 | 0.92 | 0.80 | 27.8 | 33 | 112.7 | 1314 |
| qwen3-emb-0.6B-q8 + qwen3-rerank-0.6B-q8 | 0.92 | 0.96 | 1.00 | 0.95 | 437.9 | 18245 |  |  |
| embeddinggemma-300M-q4 + qwen3-rerank-0.6B-q8 | 0.92 | 0.96 | 1.00 | 0.95 | 397.9 | 16581 |  |  |
| bge-m3-q4km + qwen3-rerank-0.6B-q8 | 0.92 | 0.96 | 1.00 | 0.95 | 360.2 | 15007 |  |  |
| [qwen3-rerank-0.6B-q8 total server] |  |  |  |  |  |  | 4426.1 | 2638 |
| qwen3-emb-0.6B-q8 + bge-reranker-v2-m3-q4km | 0.92 | 1.00 | 1.00 | 0.95 | 213.0 | 8877 |  |  |
| embeddinggemma-300M-q4 + bge-reranker-v2-m3-q4km | 0.92 | 1.00 | 1.00 | 0.95 | 244.8 | 10201 |  |  |
| bge-m3-q4km + bge-reranker-v2-m3-q4km | 0.92 | 1.00 | 1.00 | 0.95 | 227.0 | 9458 |  |  |
| [bge-reranker-v2-m3-q4km total server] |  |  |  |  |  |  | 2659.0 | 1329 |
