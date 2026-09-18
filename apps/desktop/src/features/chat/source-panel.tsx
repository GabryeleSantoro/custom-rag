import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import {
  BookOpenIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  FileTextIcon,
  FolderIcon,
  FolderPlusIcon,
  SearchIcon,
  XIcon,
} from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { IconTooltip } from "@/components/ui/tooltip";
import { AddSourceDialog } from "@/features/library/add-source-dialog";
import { relativeTime, shortPath } from "@/lib/format";
import {
  api,
  type ChatProject,
  type Citation,
  type Document,
  type RetrievedChunk,
  type Source,
} from "@/lib/ipc";
import { keys, sourcesQuery } from "@/lib/queries";
import { cn } from "@/lib/utils";

/**
 * The right-hand workspace for project resources and answer provenance.
 * Resource controls live here so they stay next to the chat they affect.
 */
export function SourcePanel({
  project,
  selectedDocumentIds,
  onSelectionChange,
  chunks,
  citations,
  selectedChunkId,
  onSelect,
  onClose,
}: {
  project: ChatProject | null;
  selectedDocumentIds: string[] | null;
  onSelectionChange: (documentIds: string[]) => void;
  chunks: RetrievedChunk[];
  citations: Citation[];
  selectedChunkId: string | null;
  onSelect: (chunkId: string) => void;
  onClose: () => void;
}) {
  const sources = useQuery(sourcesQuery);
  const documents = useQuery({
    queryKey: keys.documents({ limit: 500 }),
    queryFn: () => api.listDocuments({ limit: 500 }),
  });
  const queryClient = useQueryClient();
  const [globalOpen, setGlobalOpen] = useState(true);
  const [projectOpen, setProjectOpen] = useState(true);
  const [knowledgeSearch, setKnowledgeSearch] = useState("");
  const citedChunks = new Map(citations.map((citation) => [citation.chunk_id, citation]));

  const updateProject = useMutation({
    mutationFn: (useGlobalSources: boolean) => {
      if (!project) throw new Error("No project selected");
      return api.updateProject(project.id, { use_global_sources: useGlobalSources });
    },
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: keys.projects }),
    onError: (error: Error) =>
      toast.error("Could not update project resources", { description: error.message }),
  });

  const allDocuments = documents.data?.items ?? [];
  const globalSources = (sources.data ?? []).filter((source) => source.project_id === null);
  const globalSourceIds = useMemo(
    () => new Set(globalSources.map((source) => source.id)),
    [globalSources],
  );
  const selectableSourceIds = useMemo(
    () =>
      new Set(
        (sources.data ?? [])
          .filter((source) => !project || source.project_id === null || source.project_id === project.id)
          .map((source) => source.id),
      ),
    [project, sources.data],
  );
  const selectableDocuments = allDocuments.filter((document) =>
    selectableSourceIds.has(document.source_id),
  );
  const selectedSet = useMemo(
    () => (selectedDocumentIds === null ? null : new Set(selectedDocumentIds)),
    [selectedDocumentIds],
  );
  const isSelected = (document: Document) =>
    (!globalSourceIds.has(document.source_id) || !project || project.use_global_sources) &&
    (selectedSet === null || selectedSet.has(document.id));
  const toggleDocuments = (items: Document[], checked: boolean) => {
    const next = new Set(selectedSet ?? selectableDocuments.map((document) => document.id));
    for (const document of items) {
      if (checked) next.add(document.id);
      else next.delete(document.id);
    }
    onSelectionChange([...next]);
  };
  const projectSources = (sources.data ?? []).filter(
    (source) => source.project_id === project?.id,
  );
  const normalizedSearch = knowledgeSearch.trim().toLowerCase();
  const documentsForSource = (source: Source) =>
    allDocuments.filter((document) => document.source_id === source.id);
  const visibleDocumentsForSource = (source: Source) => {
    const folderMatches = source.path.toLowerCase().includes(normalizedSearch);
    return documentsForSource(source).filter(
      (document) =>
        !normalizedSearch ||
        folderMatches ||
        document.title.toLowerCase().includes(normalizedSearch) ||
        document.path.toLowerCase().includes(normalizedSearch),
    );
  };
  const globalDocuments = globalSources.flatMap(documentsForSource);
  const globalEnabled = !project || project.use_global_sources;
  const globalSelectedCount = globalDocuments.filter(isSelected).length;
  const globalChecked =
    globalEnabled &&
    globalDocuments.length > 0 &&
    globalSelectedCount === globalDocuments.length;
  const globalIndeterminate = globalEnabled && globalSelectedCount > 0 && !globalChecked;

  return (
    <aside className="flex w-90 shrink-0 flex-col border-l border-border bg-sidebar">
      <header
        data-tauri-drag-region="deep"
        className="drag-region flex items-center justify-between gap-2 px-3 pt-3 pb-2"
      >
        <div>
          <h2 className="text-[0.8125rem] font-semibold tracking-tight">Resources</h2>
          <p className="text-[0.6875rem] text-muted-foreground">
            {project ? project.name : "Global library"}
          </p>
        </div>
        <IconTooltip label="Close resources">
          <Button
            data-tauri-drag-region="false"
            variant="ghost"
            size="icon"
            className="no-drag size-7"
            aria-label="Close resources"
            onClick={onClose}
          >
            <XIcon className="size-4" />
          </Button>
        </IconTooltip>
      </header>

      <ScrollArea className="min-h-0 flex-1">
        <div className="space-y-4 px-3 pb-4">
          <section className="space-y-1">
            <div className="flex items-center justify-between gap-2">
              <div>
                <h3 className="text-xs font-semibold tracking-tight">Knowledge</h3>
                <p className="text-[0.6875rem] text-muted-foreground">
                  {project ? "What this project can search" : "Shared across projects"}
                </p>
              </div>
              {!project ? (
                <AddSourceDialog
                  trigger={
                    <Button variant="ghost" size="sm" className="h-7 px-2 text-xs">
                      <FolderPlusIcon className="size-3.5" />
                      Add folder
                    </Button>
                  }
                />
              ) : null}
            </div>

            <div className="relative">
              <SearchIcon className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
              <Input
                value={knowledgeSearch}
                onChange={(event) => setKnowledgeSearch(event.target.value)}
                placeholder="Search knowledge…"
                aria-label="Search knowledge"
                className="h-8 pl-8 text-xs"
              />
            </div>

            <Collapsible open={globalOpen} onOpenChange={setGlobalOpen}>
              <div className="flex items-center justify-between rounded-md border border-border bg-muted/30 px-2 py-1.5">
                <CollapsibleTrigger asChild>
                  <button
                    type="button"
                    className="flex min-w-0 items-center gap-2 text-left"
                    aria-label={`${globalOpen ? "Collapse" : "Expand"} global knowledge`}
                  >
                    {globalOpen ? (
                      <ChevronDownIcon className="size-3.5 shrink-0 text-muted-foreground" />
                    ) : (
                      <ChevronRightIcon className="size-3.5 shrink-0 text-muted-foreground" />
                    )}
                    <FolderIcon className="size-4 shrink-0 text-primary" />
                    <span className="truncate text-xs font-medium">Global knowledge</span>
                    <span className="text-[0.6875rem] text-muted-foreground">
                      {globalSources.length}
                    </span>
                  </button>
                </CollapsibleTrigger>
                <Checkbox
                  checked={globalIndeterminate ? "indeterminate" : globalChecked}
                  onCheckedChange={(checked) => {
                    const include = checked === true;
                    toggleDocuments(globalDocuments, include);
                    if (project && include !== project.use_global_sources) {
                      updateProject.mutate(include);
                    }
                  }}
                  disabled={updateProject.isPending}
                  aria-label="Include all global knowledge files"
                />
              </div>
              <CollapsibleContent>
                <div className="mt-1 space-y-0.5 pl-7">
                  {globalSources.length ? (
                    globalSources.some((source) => visibleDocumentsForSource(source).length) ? (
                      globalSources.map((source) => (
                        <FolderGroup
                          key={source.id}
                          source={source}
                          files={documentsForSource(source)}
                          visibleFiles={visibleDocumentsForSource(source)}
                          selected={isSelected}
                          onToggle={toggleDocuments}
                          disabled={Boolean(project && !project.use_global_sources)}
                        />
                      ))
                    ) : (
                      <p className="py-2 text-xs text-muted-foreground">No matching files.</p>
                    )
                  ) : (
                    <p className="py-2 text-xs text-muted-foreground">No global files yet.</p>
                  )}
                </div>
              </CollapsibleContent>
            </Collapsible>

            {project ? (
              <Collapsible open={projectOpen} onOpenChange={setProjectOpen}>
                <div className="mt-2 flex items-center justify-between rounded-md border border-border bg-muted/30 px-2 py-1.5">
                  <CollapsibleTrigger asChild>
                    <button
                      type="button"
                      className="flex min-w-0 items-center gap-2 text-left"
                      aria-label={`${projectOpen ? "Collapse" : "Expand"} project folders`}
                    >
                      {projectOpen ? (
                        <ChevronDownIcon className="size-3.5 shrink-0 text-muted-foreground" />
                      ) : (
                        <ChevronRightIcon className="size-3.5 shrink-0 text-muted-foreground" />
                      )}
                      <FolderIcon className="size-4 shrink-0 text-primary" />
                      <span className="truncate text-xs font-medium">Project folders</span>
                      <span className="text-[0.6875rem] text-muted-foreground">
                        {projectSources.length}
                      </span>
                    </button>
                  </CollapsibleTrigger>
                  <AddSourceDialog
                    projectId={project.id}
                    trigger={
                      <Button
                        variant="ghost"
                        size="icon"
                        className="size-6"
                        aria-label="Add project folder"
                      >
                        <FolderPlusIcon className="size-3.5" />
                      </Button>
                    }
                    triggerLabel="Add project folder"
                  />
                </div>
                <CollapsibleContent>
                  <div className="mt-1 space-y-0.5 pl-7">
                    {projectSources.length ? (
                      projectSources.some((source) => visibleDocumentsForSource(source).length) ? (
                        projectSources.map((source) => (
                          <FolderGroup
                            key={source.id}
                            source={source}
                            files={documentsForSource(source)}
                            visibleFiles={visibleDocumentsForSource(source)}
                            selected={isSelected}
                            onToggle={toggleDocuments}
                          />
                        ))
                      ) : (
                        <p className="py-2 text-xs text-muted-foreground">No matching files.</p>
                      )
                    ) : (
                      <p className="py-2 text-xs text-muted-foreground">No project files yet.</p>
                    )}
                  </div>
                </CollapsibleContent>
              </Collapsible>
            ) : null}
          </section>

          <section className="border-t border-border pt-3">
            <div className="mb-2 flex items-center justify-between">
              <div>
                <h3 className="text-xs font-semibold tracking-tight">Answer sources</h3>
                <p className="text-[0.6875rem] text-muted-foreground">
                  {chunks.length} passage{chunks.length === 1 ? "" : "s"} sent to the model
                </p>
              </div>
            </div>

            {chunks.length === 0 ? (
              <p className="py-5 text-center text-xs text-muted-foreground">
                Ask a question to see the passages used in the answer.
              </p>
            ) : null}

            <div className="space-y-2">
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
                      <span title="Rank in the dense list">dense {chunk.dense_rank ?? "—"}</span>
                      <span title="Rank in the keyword list">bm25 {chunk.bm25_rank ?? "—"}</span>
                    </div>

                    <Button
                      asChild
                      variant="ghost"
                      size="sm"
                      className="mt-2 h-7 w-full justify-start px-2"
                    >
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
          </section>
        </div>
      </ScrollArea>
    </aside>
  );
}

function FolderGroup({
  source,
  files,
  visibleFiles,
  selected,
  onToggle,
  disabled = false,
}: {
  source: Source;
  files: Document[];
  visibleFiles: Document[];
  selected: (document: Document) => boolean;
  onToggle: (documents: Document[], checked: boolean) => void;
  disabled?: boolean;
}) {
  const [open, setOpen] = useState(true);
  const selectedCount = files.filter(selected).length;
  const checked = !disabled && files.length > 0 && selectedCount === files.length;
  const indeterminate = !disabled && selectedCount > 0 && !checked;

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <div className={cn("flex items-center gap-1 rounded-md px-1 py-1", disabled && "opacity-50")}>
        <Checkbox
          checked={indeterminate ? "indeterminate" : checked}
          onCheckedChange={(value) => onToggle(files, value === true)}
          disabled={disabled}
          aria-label={`Include all files in ${shortPath(source.path)}`}
        />
        <CollapsibleTrigger asChild>
          <button type="button" className="flex min-w-0 flex-1 items-center gap-2 text-left">
            {open ? (
              <ChevronDownIcon className="size-3.5 shrink-0 text-muted-foreground" />
            ) : (
              <ChevronRightIcon className="size-3.5 shrink-0 text-muted-foreground" />
            )}
            <FolderIcon className="size-3.5 shrink-0 text-muted-foreground" />
            <div className="min-w-0 flex-1">
              <p className="truncate text-xs font-medium">{shortPath(source.path)}</p>
              <p className="truncate text-[0.6875rem] text-muted-foreground">
                {selectedCount}/{files.length} files included · {source.indexed_count}/
                {source.document_count} indexed
              </p>
            </div>
          </button>
        </CollapsibleTrigger>
      </div>
      <CollapsibleContent>
        <div className="ml-6 space-y-0.5 border-l border-border pl-2">
          {visibleFiles.length ? (
            visibleFiles.map((document) => (
              <FileRow
                key={document.id}
                document={document}
                selected={selected(document)}
                disabled={disabled}
                onToggle={(checked) => onToggle([document], checked)}
              />
            ))
          ) : (
            <p className="py-1 text-[0.6875rem] text-muted-foreground">No matching files.</p>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}

function FileRow({
  document,
  selected,
  disabled,
  onToggle,
}: {
  document: Document;
  selected: boolean;
  disabled: boolean;
  onToggle: (checked: boolean) => void;
}) {
  return (
    <div
      className={cn(
        "flex items-center gap-2 rounded-md px-2 py-1.5 transition-colors hover:bg-sidebar-accent/60",
        disabled && "cursor-not-allowed opacity-45",
      )}
    >
      <Checkbox
        checked={selected}
        onCheckedChange={(value) => onToggle(value === true)}
        disabled={disabled}
        aria-label={`Include ${document.title}`}
      />
      <FileTextIcon className="size-3.5 shrink-0 text-muted-foreground" />
      <div className="min-w-0 flex-1">
        <p className="truncate text-xs font-medium">{document.title}</p>
        <p className="truncate text-[0.6875rem] text-muted-foreground">
          {document.status} · {relativeTime(document.mtime)}
        </p>
      </div>
    </div>
  );
}
