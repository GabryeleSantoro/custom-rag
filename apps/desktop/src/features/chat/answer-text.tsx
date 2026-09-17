import { Fragment } from "react";

import type { Citation } from "@/lib/ipc";
import { cn } from "@/lib/utils";

/** Matches the [document_id:page] markers the prompt asks the model to write. */
const MARKER = /\[([A-Za-z0-9_.\-]+):(\d+)\]/g;

export type CitationTarget = { docId: string; page: number; citation?: Citation };

function Chip({
  docId,
  page,
  index,
  known,
  onSelect,
}: {
  docId: string;
  page: number;
  index: number | null;
  known: boolean;
  onSelect: () => void;
}) {
  const label = index ?? "?";
  return (
    <button
      type="button"
      onClick={onSelect}
      title={known ? `${docId}, page ${page}` : `${docId} was not among the retrieved passages`}
      className={cn(
        "mx-0.5 inline-flex h-4.5 min-w-4.5 translate-y-[-1px] items-center justify-center",
        "rounded-[0.25rem] px-1 align-middle font-mono text-[0.625rem] font-medium tabular-nums",
        "transition-colors",
        known
          ? "bg-primary/12 text-primary hover:bg-primary/22"
          : "bg-destructive/12 text-destructive line-through",
      )}
    >
      {label}
    </button>
  );
}

/**
 * Renders an answer: light markdown (paragraphs, bullets) plus citation chips.
 *
 * A full markdown renderer is deliberately not pulled in yet — the scripted
 * answers are plain, and the pieces that matter here are the chips and the
 * distinction between a citation that resolves and one that does not.
 */
export function AnswerText({
  text,
  citations,
  onSelect,
  className,
}: {
  text: string;
  citations: Citation[];
  onSelect?: (target: CitationTarget) => void;
  className?: string;
}) {
  const order = new Map<string, number>();
  citations.forEach((citation, index) => order.set(citation.marker, index + 1));

  const renderInline = (line: string, keyPrefix: string) => {
    const nodes: React.ReactNode[] = [];
    let cursor = 0;
    MARKER.lastIndex = 0;

    for (let match = MARKER.exec(line); match; match = MARKER.exec(line)) {
      if (match.index > cursor) nodes.push(line.slice(cursor, match.index));
      const [marker, docId, pageRaw] = match;
      const page = Number(pageRaw);
      const index = order.get(marker) ?? null;
      const citation = citations.find((c) => c.marker === marker);
      nodes.push(
        <Chip
          key={`${keyPrefix}-${match.index}`}
          docId={docId}
          page={page}
          index={index}
          known={Boolean(citation)}
          onSelect={() => onSelect?.({ docId, page, citation })}
        />,
      );
      cursor = match.index + marker.length;
    }
    if (cursor < line.length) nodes.push(line.slice(cursor));
    return nodes;
  };

  const blocks = text.split("\n");

  return (
    <div className={cn("selectable space-y-2 text-[0.875rem] leading-[1.6]", className)}>
      {blocks.map((line, index) => {
        const trimmed = line.trim();
        if (!trimmed) return null;
        if (trimmed.startsWith("- ")) {
          return (
            <div key={index} className="flex gap-2 pl-1">
              <span className="mt-[0.55em] size-1 shrink-0 rounded-full bg-muted-foreground" />
              <p className="flex-1">{renderInline(trimmed.slice(2), `b${index}`)}</p>
            </div>
          );
        }
        return (
          <p key={index}>
            <Fragment>{renderInline(trimmed, `p${index}`)}</Fragment>
          </p>
        );
      })}
    </div>
  );
}
