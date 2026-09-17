import { useState } from "react";
import { Link } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { BookOpenIcon, RefreshCwIcon, SearchIcon, Trash2Icon } from "lucide-react";
import { toast } from "sonner";

import { Page, PageBody, PageHeader } from "@/components/shell/page";
import { StatusChip, documentTone } from "@/components/status";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import { SourceSidebar } from "@/features/library/source-sidebar";
import { bytes, relativeTime } from "@/lib/format";
import { api } from "@/lib/ipc";
import { useJobs } from "@/lib/jobs-context";
import { keys } from "@/lib/queries";

export function LibraryView() {
  const queryClient = useQueryClient();
  const [sourceId, setSourceId] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const { active } = useJobs();

  const params = { source_id: sourceId ?? undefined, q: search || undefined, limit: 500 };
  const documents = useQuery({
    queryKey: keys.documents(params),
    queryFn: () => api.listDocuments(params),
  });

  const invalidate = () => {
    void queryClient.invalidateQueries({ queryKey: ["documents"] });
    void queryClient.invalidateQueries({ queryKey: keys.sources });
  };

  const reindex = useMutation({
    mutationFn: (docIds: string[]) => api.rebuild({ doc_ids: docIds }),
    onSuccess: () => toast.success("Re-indexing started"),
    onError: (error: Error) => toast.error("Re-index failed", { description: error.message }),
  });

  const remove = useMutation({
    mutationFn: (id: string) => api.removeDocument(id),
    onSuccess: invalidate,
    onError: (error: Error) => toast.error("Could not remove", { description: error.message }),
  });

  const items = documents.data?.items ?? [];
  const errored = items.filter((document) => document.status === "error");

  return (
    <>
      <SourceSidebar selected={sourceId} onSelect={setSourceId} />

      <Page>
        <PageHeader
          title="Library"
          description={
            documents.data
              ? `${documents.data.total} document${documents.data.total === 1 ? "" : "s"}${
                  active.length ? ` · ${active.length} job${active.length === 1 ? "" : "s"} running` : ""
                }`
              : "Loading…"
          }
          actions={
            <>
              <div className="relative">
                <SearchIcon className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
                <Input
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                  placeholder="Filter by title or path"
                  className="h-8 w-56 pl-8 text-xs"
                />
              </div>
              <Button
                variant="secondary"
                size="sm"
                className="h-8"
                disabled={items.length === 0 || reindex.isPending}
                onClick={() => reindex.mutate(items.map((document) => document.id))}
              >
                <RefreshCwIcon className="size-3.5" />
                Re-index all
              </Button>
            </>
          }
        />

        <PageBody>
          {errored.length > 0 ? (
            <div className="border-b border-status-error/25 bg-status-error/8 px-5 py-2 text-xs text-status-error">
              {errored.length} document{errored.length === 1 ? "" : "s"} failed to index.
              {errored[0].error ? ` First error: ${errored[0].error}` : ""}
            </div>
          ) : null}

          {documents.isLoading ? (
            <div className="space-y-2 p-5">
              <Skeleton className="h-9 w-full" />
              <Skeleton className="h-9 w-full" />
              <Skeleton className="h-9 w-full" />
            </div>
          ) : items.length === 0 ? (
            <div className="grid h-full place-items-center p-10 text-center">
              <div className="max-w-sm">
                <p className="text-sm font-medium">Nothing indexed yet</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Add a source folder from the sidebar. Files are parsed and embedded on this
                  machine.
                </p>
              </div>
            </div>
          ) : (
            <Table>
              <TableHeader className="sticky top-0 z-10 bg-background">
                <TableRow>
                  <TableHead className="w-[42%]">Document</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">Pages</TableHead>
                  <TableHead className="text-right">Passages</TableHead>
                  <TableHead className="text-right">Size</TableHead>
                  <TableHead className="text-right">Modified</TableHead>
                  <TableHead className="w-24" />
                </TableRow>
              </TableHeader>
              <TableBody>
                {items.map((document) => (
                  <TableRow key={document.id} className="group">
                    <TableCell className="max-w-0">
                      <p className="truncate text-[0.8125rem] font-medium">{document.title}</p>
                      <p className="truncate font-mono text-[0.6875rem] text-muted-foreground">
                        {document.path}
                      </p>
                    </TableCell>
                    <TableCell>
                      <StatusChip
                        tone={documentTone(document.status)}
                        label={document.status}
                        pulse={!["indexed", "error", "skipped"].includes(document.status)}
                      />
                    </TableCell>
                    <TableCell className="text-right font-mono text-xs tabular-nums">
                      {document.n_pages ?? "—"}
                    </TableCell>
                    <TableCell className="text-right font-mono text-xs tabular-nums">
                      {document.n_chunks}
                    </TableCell>
                    <TableCell className="text-right font-mono text-xs tabular-nums">
                      {bytes(document.size_bytes)}
                    </TableCell>
                    <TableCell className="text-right text-xs whitespace-nowrap text-muted-foreground">
                      {relativeTime(document.mtime)}
                    </TableCell>
                    <TableCell>
                      <div className="flex justify-end gap-0.5 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
                        <Button asChild variant="ghost" size="icon" className="size-7">
                          <Link
                            to="/reader/$docId"
                            params={{ docId: document.id }}
                            aria-label={`Open ${document.title}`}
                          >
                            <BookOpenIcon className="size-3.5" />
                          </Link>
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="size-7"
                          aria-label="Re-index"
                          onClick={() => reindex.mutate([document.id])}
                        >
                          <RefreshCwIcon className="size-3.5" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="size-7"
                          aria-label="Remove"
                          onClick={() => remove.mutate(document.id)}
                        >
                          <Trash2Icon className="size-3.5" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </PageBody>
      </Page>
    </>
  );
}
