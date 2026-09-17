import { useEffect, useMemo, useRef } from "react";
import { Link, useNavigate, useParams, useSearch } from "@tanstack/react-router";
import { useMutation, useQuery } from "@tanstack/react-query";
import { ChevronLeftIcon, ChevronRightIcon, MessageSquareIcon } from "lucide-react";
import { toast } from "sonner";

import { ContextSidebar } from "@/components/shell/context-sidebar";
import { Page, PageBody, PageHeader } from "@/components/shell/page";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { bytes, relativeTime } from "@/lib/format";
import { api, type DocumentChunkRef } from "@/lib/ipc";
import { keys } from "@/lib/queries";
import { cn } from "@/lib/utils";

/** Splits a page into plain text and the highlighted span, if one falls here. */
function segments(text: string, chunk: DocumentChunkRef | undefined) {
  if (!chunk) return [{ text, mark: false }];
  const start = Math.max(0, Math.min(chunk.char_start, text.length));
  const end = Math.max(start, Math.min(chunk.char_end, text.length));
  return [
    { text: text.slice(0, start), mark: false },
    { text: text.slice(start, end), mark: true },
    { text: text.slice(end), mark: false },
  ].filter((segment) => segment.text.length > 0);
}

export function ReaderView() {
  const { docId } = useParams({ from: "/_shell/reader/$docId" });
  const search = useSearch({ from: "/_shell/reader/$docId" });
  const navigate = useNavigate();
  const marked = useRef<HTMLElement>(null);

  const document = useQuery({
    queryKey: keys.document(docId),
    queryFn: () => api.getDocument(docId),
  });
  const content = useQuery({
    queryKey: keys.content(docId),
    queryFn: () => api.getContent(docId),
  });

  const highlighted = useMemo(
    () => content.data?.chunks.find((chunk) => chunk.chunk_id === search.highlight),
    [content.data, search.highlight],
  );

  const currentPage = search.page ?? highlighted?.page ?? 1;
  const pages = content.data?.pages ?? [];
  const page = pages.find((entry) => entry.page === currentPage) ?? pages[0];

  useEffect(() => {
    if (highlighted && page?.page === highlighted.page) {
      marked.current?.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [highlighted, page]);

  const goTo = (target: number) =>
    navigate({
      to: "/reader/$docId",
      params: { docId },
      search: { page: target, highlight: search.highlight },
    });

  const askAbout = useMutation({
    mutationFn: () =>
      api.createSession({
        title: `About ${document.data?.title ?? docId}`,
        scope_doc_id: docId,
      }),
    onSuccess: (session) =>
      navigate({ to: "/chat/$sessionId", params: { sessionId: session.id } }),
    onError: (error: Error) => toast.error("Could not start a chat", { description: error.message }),
  });

  return (
    <>
      <ContextSidebar title="Pages">
        {content.isLoading ? <Skeleton className="h-40 w-full" /> : null}
        <nav className="flex flex-col gap-0.5 pt-1">
          {pages.map((entry) => {
            const hasHighlight = highlighted?.page === entry.page;
            return (
              <button
                key={entry.page}
                type="button"
                onClick={() => goTo(entry.page)}
                className={cn(
                  "rounded-md px-2 py-1.5 text-left transition-colors",
                  entry.page === currentPage
                    ? "bg-sidebar-accent"
                    : "hover:bg-sidebar-accent/60",
                )}
              >
                <div className="flex items-baseline gap-2">
                  <span className="font-mono text-[0.625rem] text-muted-foreground tabular-nums">
                    {String(entry.page).padStart(2, "0")}
                  </span>
                  <span
                    className={cn(
                      "flex-1 truncate text-[0.8125rem]",
                      entry.page === currentPage && "font-medium",
                    )}
                  >
                    {entry.section_path?.split(" > ").at(-1) ?? `Page ${entry.page}`}
                  </span>
                  {hasHighlight ? <span className="size-1.5 rounded-full bg-primary" /> : null}
                </div>
              </button>
            );
          })}
        </nav>
      </ContextSidebar>

      <Page>
        <PageHeader
          actions={
            <>
              <Button
                variant="secondary"
                size="sm"
                className="h-8"
                disabled={askAbout.isPending}
                onClick={() => askAbout.mutate()}
              >
                <MessageSquareIcon className="size-3.5" />
                Ask about this document
              </Button>
              <Button asChild variant="ghost" size="sm" className="h-8">
                <Link to="/library">Library</Link>
              </Button>
            </>
          }
        >
          <div className="min-w-0 flex-1">
            <h1 className="truncate text-sm font-semibold tracking-tight">
              {document.data?.title ?? "Loading…"}
            </h1>
            <p className="truncate font-mono text-[0.6875rem] text-muted-foreground">
              {document.data
                ? `${document.data.path} · ${bytes(document.data.size_bytes)} · ${relativeTime(document.data.mtime)}`
                : ""}
            </p>
          </div>
        </PageHeader>

        <PageBody>
          {content.isLoading ? (
            <div className="mx-auto max-w-2xl space-y-3 p-10">
              <Skeleton className="h-6 w-1/2" />
              <Skeleton className="h-40 w-full" />
            </div>
          ) : !page ? (
            <div className="grid h-full place-items-center text-sm text-muted-foreground">
              This document has no extractable text.
            </div>
          ) : (
            <article className="mx-auto max-w-2xl px-10 py-10">
              <p className="mb-6 font-mono text-[0.6875rem] tracking-wide text-muted-foreground uppercase">
                {page.section_path ?? `Page ${page.page}`}
              </p>
              <div className="selectable font-reader text-[0.9375rem] leading-[1.75] whitespace-pre-wrap">
                {segments(page.text, highlighted?.page === page.page ? highlighted : undefined).map(
                  (segment, index) =>
                    segment.mark ? (
                      <mark
                        key={index}
                        ref={marked}
                        className="rounded-sm bg-primary/18 px-0.5 py-px text-foreground"
                      >
                        {segment.text}
                      </mark>
                    ) : (
                      <span key={index}>{segment.text}</span>
                    ),
                )}
              </div>
            </article>
          )}
        </PageBody>

        <footer className="flex items-center justify-between border-t border-border px-5 py-2">
          <Button
            variant="ghost"
            size="sm"
            className="h-7"
            disabled={currentPage <= 1}
            onClick={() => goTo(currentPage - 1)}
          >
            <ChevronLeftIcon className="size-3.5" />
            Previous
          </Button>
          <span className="font-mono text-[0.6875rem] text-muted-foreground tabular-nums">
            {currentPage} / {content.data?.n_pages ?? "?"}
          </span>
          <Button
            variant="ghost"
            size="sm"
            className="h-7"
            disabled={currentPage >= (content.data?.n_pages ?? 1)}
            onClick={() => goTo(currentPage + 1)}
          >
            Next
            <ChevronRightIcon className="size-3.5" />
          </Button>
        </footer>
      </Page>
    </>
  );
}
