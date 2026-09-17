import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { CloudIcon, FileTextIcon, PanelRightIcon } from "lucide-react";

import { Page, PageBody, PageHeader } from "@/components/shell/page";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Composer } from "@/features/chat/composer";
import { AssistantMessage, UserMessage } from "@/features/chat/message";
import { SourcePanel } from "@/features/chat/source-panel";
import { useChat } from "@/features/chat/use-chat";
import type { CitationTarget } from "@/features/chat/answer-text";
import { api, type ChatMessage, type QueryMode } from "@/lib/ipc";
import { connectionsQuery, keys, sourcesQuery } from "@/lib/queries";

const SUGGESTIONS = [
  "How many candidates should the reranker get?",
  "Why can't the embedder be swapped without re-indexing?",
  "What does reciprocal rank fusion actually combine?",
];

export function ChatView({ sessionId }: { sessionId: string | null }) {
  const navigate = useNavigate();
  const [mode, setMode] = useState<QueryMode>("auto");
  const [sourceId, setSourceId] = useState<string>("all");
  const [panelOpen, setPanelOpen] = useState(true);
  const [selectedChunk, setSelectedChunk] = useState<string | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  const { pending, isStreaming, send, cancel, activeSession } = useChat(sessionId);
  const sources = useQuery(sourcesQuery);
  const connections = useQuery(connectionsQuery);

  const session = useQuery({
    queryKey: ["session", sessionId],
    queryFn: () => api.getSession(sessionId as string),
    enabled: Boolean(sessionId),
  });

  const history = useQuery({
    queryKey: keys.messages(sessionId ?? "none"),
    queryFn: () => api.getMessages(sessionId as string),
    enabled: Boolean(sessionId),
  });

  const active = connections.data?.find((connection) => connection.active);

  // A new chat gets its id from the first `start` frame. Move to its URL once
  // the stream is over, so the history query takes over from the live state.
  useEffect(() => {
    if (!sessionId && activeSession && pending && pending.status !== "retrieving" && pending.status !== "streaming") {
      void navigate({ to: "/chat/$sessionId", params: { sessionId: activeSession } });
    }
  }, [activeSession, navigate, pending, sessionId]);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: pending ? "smooth" : "auto", block: "end" });
  }, [pending?.text, history.data?.length, pending]);

  const lastChunks = useMemo(() => {
    if (pending?.chunks.length) return pending.chunks;
    const lastAssistant = [...(history.data ?? [])].reverse().find((m) => m.role === "assistant");
    return lastAssistant?.chunks ?? [];
  }, [history.data, pending]);

  const lastCitations = useMemo(() => {
    if (pending?.citations.length) return pending.citations;
    const lastAssistant = [...(history.data ?? [])].reverse().find((m) => m.role === "assistant");
    return lastAssistant?.citations ?? [];
  }, [history.data, pending]);

  const onSelectCitation = (target: CitationTarget) => {
    if (!target.citation) return;
    setPanelOpen(true);
    setSelectedChunk(target.citation.chunk_id);
    requestAnimationFrame(() => {
      document
        .getElementById(`source-${target.citation!.chunk_id}`)
        ?.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  };

  const scopeDocId = session.data?.scope_doc_id ?? null;

  const onSend = (text: string) => {
    send(text, {
      mode,
      filters: scopeDocId
        ? { doc_ids: [scopeDocId] }
        : sourceId === "all"
          ? {}
          : { source_ids: [sourceId] },
    });
  };

  return (
    <>
      <Page>
        <PageHeader
          title={sessionId ? undefined : "New chat"}
          actions={
            <Button
              variant="ghost"
              size="icon"
              className="size-8"
              aria-pressed={panelOpen}
              onClick={() => setPanelOpen((open) => !open)}
            >
              <PanelRightIcon className="size-4" />
              <span className="sr-only">Toggle sources</span>
            </Button>
          }
        >
          <div className="min-w-0 flex-1">
            <h1 className="truncate text-sm font-semibold tracking-tight">
              {sessionId ? "Chat" : "New chat"}
            </h1>
            <p className="truncate text-xs text-muted-foreground">
              {active ? `${active.name} · ${active.model_id}` : "No model connected"}
            </p>
          </div>
        </PageHeader>

        {active?.is_remote ? (
          <div className="flex items-center gap-2 border-b border-status-warn/25 bg-status-warn/8 px-5 py-1.5 text-[0.6875rem] text-status-warn">
            <CloudIcon className="size-3.5 shrink-0" />
            Retrieved passages leave this device: they are sent to {active.name}. Embedding and
            reranking stay local.
          </div>
        ) : null}

        <PageBody>
          <div className="mx-auto w-full max-w-3xl space-y-6 px-5 py-6">
            {history.isLoading ? (
              <div className="space-y-3">
                <Skeleton className="h-8 w-2/3" />
                <Skeleton className="h-24 w-full" />
              </div>
            ) : null}

            {(history.data ?? []).map((message: ChatMessage) =>
              message.role === "user" ? (
                <UserMessage key={message.id} text={message.text} />
              ) : (
                <AssistantMessage
                  key={message.id}
                  text={message.text}
                  mode={message.mode ?? null}
                  citations={message.citations ?? []}
                  chunks={message.chunks ?? []}
                  grounding={message.grounding ?? null}
                  dropped={0}
                  latency={null}
                  onSelectCitation={onSelectCitation}
                />
              ),
            )}

            {pending ? (
              <>
                <UserMessage text={pending.question} />
                <AssistantMessage
                  text={pending.text}
                  mode={pending.mode}
                  modeReason={pending.modeReason}
                  citations={pending.citations}
                  chunks={pending.chunks}
                  grounding={pending.grounding}
                  dropped={pending.dropped}
                  latency={pending.latency}
                  streaming={pending.status === "streaming"}
                  retrieving={pending.status === "retrieving"}
                  error={pending.error}
                  onSelectCitation={onSelectCitation}
                />
              </>
            ) : null}

            {!pending && !sessionId ? (
              <div className="pt-10">
                <h2 className="text-lg font-semibold tracking-tight">Ask your documents</h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  Answers are built from passages in your library, and every claim links back to
                  the page it came from.
                </p>
                <div className="mt-5 grid gap-2">
                  {SUGGESTIONS.map((suggestion) => (
                    <button
                      key={suggestion}
                      type="button"
                      onClick={() => onSend(suggestion)}
                      className="rounded-lg border border-border bg-card px-3 py-2 text-left text-[0.8125rem] transition-colors hover:border-muted-foreground/40"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            ) : null}

            <div ref={bottom} />
          </div>
        </PageBody>

        <Composer onSend={onSend} onStop={cancel} streaming={isStreaming}>
          {scopeDocId ? (
            <Badge variant="secondary" className="h-7 gap-1.5 font-normal">
              <FileTextIcon className="size-3" />
              Scoped to one document
            </Badge>
          ) : null}

          <Select value={mode} onValueChange={(value) => setMode(value as QueryMode)}>
            <SelectTrigger size="sm" className="h-7 gap-1 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="auto">Auto mode</SelectItem>
              <SelectItem value="local">Local: passages</SelectItem>
              <SelectItem value="global">Global: whole corpus</SelectItem>
            </SelectContent>
          </Select>

          <Select value={sourceId} onValueChange={setSourceId} disabled={Boolean(scopeDocId)}>
            <SelectTrigger size="sm" className="h-7 gap-1 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All sources</SelectItem>
              {(sources.data ?? []).map((source) => (
                <SelectItem key={source.id} value={source.id}>
                  {source.path.split("/").slice(-2).join("/")}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Composer>
      </Page>

      {panelOpen ? (
        <SourcePanel
          chunks={lastChunks}
          citations={lastCitations}
          selectedChunkId={selectedChunk}
          onSelect={setSelectedChunk}
          onClose={() => setPanelOpen(false)}
        />
      ) : null}
    </>
  );
}
