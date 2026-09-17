import { AlertTriangleIcon, GlobeIcon, LayersIcon, Loader2Icon } from "lucide-react";

import { AnswerText, type CitationTarget } from "@/features/chat/answer-text";
import { Badge } from "@/components/ui/badge";
import type { Citation, Grounding, RetrievedChunk, StageLatency } from "@/lib/ipc";
import { ms } from "@/lib/format";
import { cn } from "@/lib/utils";

export function UserMessage({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <div className="selectable max-w-[80%] rounded-2xl rounded-br-md bg-secondary px-3.5 py-2 text-sm leading-[1.55]">
        {text}
      </div>
    </div>
  );
}

export function ModeBadge({ mode, reason }: { mode: "local" | "global"; reason?: string | null }) {
  const Icon = mode === "global" ? GlobeIcon : LayersIcon;
  return (
    <Badge variant="outline" className="gap-1 font-normal" title={reason ?? undefined}>
      <Icon className="size-3" />
      {mode === "global" ? "Global" : "Local"}
    </Badge>
  );
}

function GroundingNotice({ grounding, dropped }: { grounding: Grounding; dropped: number }) {
  if (grounding === "ok" && dropped === 0) return null;

  const message =
    grounding === "none"
      ? "No citations. Nothing in this answer is backed by a retrieved passage."
      : dropped > 0
        ? `${dropped} citation${dropped === 1 ? "" : "s"} pointed at documents that were not retrieved and were removed.`
        : "Only one passage backs this answer. Treat it as weakly grounded.";

  return (
    <div
      className={cn(
        "mt-2 flex items-start gap-2 rounded-lg border px-2.5 py-2 text-xs",
        grounding === "none"
          ? "border-status-error/30 bg-status-error/8 text-status-error"
          : "border-status-warn/30 bg-status-warn/8 text-status-warn",
      )}
    >
      <AlertTriangleIcon className="mt-px size-3.5 shrink-0" />
      <span className="leading-[1.5]">{message}</span>
    </div>
  );
}

export function AssistantMessage({
  text,
  mode,
  modeReason,
  citations,
  chunks,
  grounding,
  dropped,
  latency,
  streaming,
  retrieving,
  error,
  onSelectCitation,
}: {
  text: string;
  mode: "local" | "global" | null;
  modeReason?: string | null;
  citations: Citation[];
  chunks: RetrievedChunk[];
  grounding: Grounding | null;
  dropped: number;
  latency: StageLatency | null;
  streaming?: boolean;
  retrieving?: boolean;
  error?: string | null;
  onSelectCitation?: (target: CitationTarget) => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        {mode ? <ModeBadge mode={mode} reason={modeReason} /> : null}
        {retrieving ? (
          <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <Loader2Icon className="size-3 animate-spin" />
            Searching {chunks.length ? `${chunks.length} passages` : "the index"}…
          </span>
        ) : null}
        {!retrieving && chunks.length > 0 ? (
          <span className="text-[0.6875rem] text-muted-foreground">
            {chunks.length} passage{chunks.length === 1 ? "" : "s"}
          </span>
        ) : null}
      </div>

      {error ? (
        <div className="rounded-lg border border-status-error/30 bg-status-error/8 px-3 py-2 text-xs text-status-error">
          {error}
        </div>
      ) : (
        <>
          <AnswerText text={text} citations={citations} onSelect={onSelectCitation} />
          {streaming ? (
            <span className="inline-block h-4 w-1.5 translate-y-0.5 animate-pulse rounded-xs bg-foreground/70" />
          ) : null}
        </>
      )}

      {grounding && !streaming && !retrieving ? (
        <GroundingNotice grounding={grounding} dropped={dropped} />
      ) : null}

      {latency && !streaming ? (
        <p className="font-mono text-[0.625rem] text-muted-foreground tabular-nums">
          retrieval {ms(latency.embed_ms + latency.dense_ms + latency.bm25_ms + latency.rerank_ms)}
          {" · "}first token {ms(latency.llm_first_token_ms)}
          {" · "}total {ms(latency.total_ms)}
        </p>
      ) : null}
    </div>
  );
}
