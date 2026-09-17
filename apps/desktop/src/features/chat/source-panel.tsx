import { Link } from "@tanstack/react-router";
import { BookOpenIcon, XIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { Citation, RetrievedChunk } from "@/lib/ipc";
import { cn } from "@/lib/utils";

/**
 * The right-hand panel. Shows every passage that reached the prompt, in the
 * order the reranker put them, with the scores that decided it — so an answer
 * can always be traced back to the text it came from.
 */
export function SourcePanel({
  chunks,
  citations,
  selectedChunkId,
  onSelect,
  onClose,
}: {
  chunks: RetrievedChunk[];
  citations: Citation[];
  selectedChunkId: string | null;
  onSelect: (chunkId: string) => void;
  onClose: () => void;
}) {
  const citedChunks = new Map(citations.map((citation) => [citation.chunk_id, citation]));

  return (
    <aside className="flex w-90 shrink-0 flex-col border-l border-border bg-sidebar">
      <header className="drag-region flex items-center justify-between gap-2 px-3 pt-3 pb-2">
        <div>
          <h2 className="text-[0.8125rem] font-semibold tracking-tight">Sources</h2>
          <p className="text-[0.6875rem] text-muted-foreground">
            {chunks.length} passage{chunks.length === 1 ? "" : "s"} sent to the model
          </p>
        </div>
        <Button variant="ghost" size="icon" className="no-drag size-7" onClick={onClose}>
          <XIcon className="size-4" />
          <span className="sr-only">Close</span>
        </Button>
      </header>

      <ScrollArea className="flex-1">
        <div className="space-y-2 px-3 pb-4">
          {chunks.length === 0 ? (
            <p className="py-8 text-center text-xs text-muted-foreground">
              No passages cleared the score gate.
            </p>
          ) : null}

          {chunks.map((chunk, index) => {
            const citation = citedChunks.get(chunk.chunk_id);
            const selected = selectedChunkId === chunk.chunk_id;
            return (
              <article
                key={chunk.chunk_id}
                id={`source-${chunk.chunk_id}`}
                onClick={() => onSelect(chunk.chunk_id)}
                className={cn(
                  "cursor-pointer rounded-lg border bg-card p-3 transition-colors",
                  selected
                    ? "border-primary/60 ring-[3px] ring-primary/15"
                    : "border-border hover:border-muted-foreground/40",
                )}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <p className="truncate text-[0.8125rem] font-medium">{chunk.doc_title}</p>
                    <p className="truncate text-[0.6875rem] text-muted-foreground">
                      {chunk.section_path ?? "—"}
                    </p>
                  </div>
                  <span
                    className={cn(
                      "grid size-5 shrink-0 place-items-center rounded font-mono text-[0.625rem]",
                      citation ? "bg-primary/12 text-primary" : "bg-muted text-muted-foreground",
                    )}
                    title={citation ? "Cited in the answer" : "Retrieved but not cited"}
                  >
                    {index + 1}
                  </span>
                </div>

                <p className="selectable mt-2 line-clamp-6 text-xs leading-[1.55] text-muted-foreground">
                  {chunk.text}
                </p>

                <div className="mt-2.5 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[0.625rem] text-muted-foreground tabular-nums">
                  <span>p.{chunk.page_start}</span>
                  <span title="Reranker score after the sigmoid">
                    rerank {chunk.rerank_score.toFixed(3)}
                  </span>
                  <span title="Rank in the dense list">
                    dense {chunk.dense_rank ?? "—"}
                  </span>
                  <span title="Rank in the keyword list">bm25 {chunk.bm25_rank ?? "—"}</span>
                </div>

                <Button asChild variant="ghost" size="sm" className="mt-2 h-7 w-full justify-start px-2">
                  <Link
                    to="/reader/$docId"
                    params={{ docId: chunk.doc_id }}
                    search={{ page: chunk.page_start, highlight: chunk.chunk_id }}
                  >
                    <BookOpenIcon className="size-3.5" />
                    Open in reader
                  </Link>
                </Button>
              </article>
            );
          })}
        </div>
      </ScrollArea>
    </aside>
  );
}
