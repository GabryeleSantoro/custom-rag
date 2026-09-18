import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ChevronDownIcon,
  ChevronRightIcon,
  FolderIcon,
  PlusIcon,
  RefreshCwIcon,
  Trash2Icon,
} from "lucide-react";
import { toast } from "sonner";

import { ContextSidebar } from "@/components/shell/context-sidebar";
import { StatusDot } from "@/components/status";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { AddSourceDialog } from "@/features/library/add-source-dialog";
import { relativeTime, shortPath } from "@/lib/format";
import { api, type Job, type Source } from "@/lib/ipc";
import { useJobs } from "@/lib/jobs-context";
import { keys, projectsQuery, sourcesQuery } from "@/lib/queries";
import { cn } from "@/lib/utils";

export function SourceSidebar({
  selected,
  onSelect,
}: {
  selected: string | null;
  onSelect: (sourceId: string | null) => void;
}) {
  const queryClient = useQueryClient();
  const sources = useQuery(sourcesQuery);
  const projects = useQuery(projectsQuery);
  const { bySource } = useJobs();
  const [globalOpen, setGlobalOpen] = useState(true);
  const [openProjects, setOpenProjects] = useState<Set<string>>(new Set());

  const invalidate = () => {
    void queryClient.invalidateQueries({ queryKey: keys.sources });
    void queryClient.invalidateQueries({ queryKey: ["documents"] });
  };

  const rescan = useMutation({
    mutationFn: (id: string) => api.rescanSource(id),
    onSuccess: invalidate,
    onError: (error: Error) => toast.error("Rescan failed", { description: error.message }),
  });

  const remove = useMutation({
    mutationFn: (id: string) => api.removeSource(id),
    onSuccess: (_result, id) => {
      if (selected === id) onSelect(null);
      invalidate();
      toast.success("Source removed");
    },
    onError: (error: Error) => toast.error("Could not remove the source", { description: error.message }),
  });

  return (
    <ContextSidebar
      title="Sources"
      action={
        <AddSourceDialog
          trigger={
            <Button variant="ghost" size="icon" className="size-7" aria-label="Add source">
              <PlusIcon className="size-4" />
            </Button>
          }
          triggerLabel="Add source"
        />
      }
    >
      <button
        type="button"
        onClick={() => onSelect(null)}
        className={cn(
          "w-full rounded-md px-2 py-1.5 text-left text-[0.8125rem] transition-colors",
          selected === null ? "bg-sidebar-accent font-medium" : "hover:bg-sidebar-accent/60",
        )}
      >
        All documents
      </button>

      {sources.isLoading ? <Skeleton className="mt-2 h-14 w-full" /> : null}

      {sources.data?.length === 0 ? (
        <div className="px-2 py-8 text-center">
          <FolderIcon className="mx-auto size-5 text-muted-foreground" strokeWidth={1.5} />
          <p className="mt-2 text-xs text-muted-foreground">
            No folders yet. Add one to start indexing.
          </p>
        </div>
      ) : null}

      <div className="mt-1 space-y-1">
        <SourceFolder
          label="Global knowledge"
          count={(sources.data ?? []).filter((source) => source.project_id === null).length}
          open={globalOpen}
          onOpenChange={setGlobalOpen}
        >
          {(sources.data ?? [])
            .filter((source) => source.project_id === null)
            .map((source) => (
              <SourceRow
                key={source.id}
                source={source}
                selected={selected}
                bySource={bySource}
                onSelect={onSelect}
                onRescan={(id) => rescan.mutate(id)}
                onRemove={(id) => remove.mutate(id)}
              />
            ))}
        </SourceFolder>

        {[...(projects.data ?? [])]
          .map((project) => ({
            project,
            items: (sources.data ?? []).filter((source) => source.project_id === project.id),
          }))
          .filter(({ items }) => items.length > 0)
          .map(({ project, items }) => {
            const open = openProjects.has(project.id);
            return (
              <SourceFolder
                key={project.id}
                label={project.name}
                count={items.length}
                open={open}
                onOpenChange={(next) =>
                  setOpenProjects((current) => {
                    const updated = new Set(current);
                    if (next) updated.add(project.id);
                    else updated.delete(project.id);
                    return updated;
                  })
                }
              >
                {items.map((source) => (
                  <SourceRow
                    key={source.id}
                    source={source}
                    selected={selected}
                    bySource={bySource}
                    onSelect={onSelect}
                    onRescan={(id) => rescan.mutate(id)}
                    onRemove={(id) => remove.mutate(id)}
                  />
                ))}
              </SourceFolder>
            );
          })}
      </div>
    </ContextSidebar>
  );
}

function SourceFolder({
  label,
  count,
  open,
  onOpenChange,
  children,
}: {
  label: string;
  count: number;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  children: React.ReactNode;
}) {
  return (
    <Collapsible open={open} onOpenChange={onOpenChange}>
      <CollapsibleTrigger asChild>
        <button
          type="button"
          className="flex w-full items-center gap-1.5 rounded-md px-2 py-1.5 text-left hover:bg-sidebar-accent/60"
          aria-label={`${open ? "Collapse" : "Expand"} ${label}`}
        >
          {open ? (
            <ChevronDownIcon className="size-3.5 shrink-0 text-muted-foreground" />
          ) : (
            <ChevronRightIcon className="size-3.5 shrink-0 text-muted-foreground" />
          )}
          <FolderIcon className="size-3.5 shrink-0 text-primary" />
          <span className="truncate text-[0.8125rem] font-medium">{label}</span>
          <span className="ml-auto text-[0.6875rem] text-muted-foreground">{count}</span>
        </button>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="mt-0.5 flex flex-col gap-0.5 pl-2">{children}</div>
      </CollapsibleContent>
    </Collapsible>
  );
}

function SourceRow({
  source,
  selected,
  bySource,
  onSelect,
  onRescan,
  onRemove,
}: {
  source: Source;
  selected: string | null;
  bySource: Map<string, Job>;
  onSelect: (sourceId: string) => void;
  onRescan: (sourceId: string) => void;
  onRemove: (sourceId: string) => void;
}) {
  const job = bySource.get(source.id);
  const active = selected === source.id;

  return (
    <div
      className={cn(
        "group rounded-md px-2 py-1.5 transition-colors",
        active ? "bg-sidebar-accent" : "hover:bg-sidebar-accent/60",
      )}
    >
      <button type="button" onClick={() => onSelect(source.id)} className="block w-full text-left">
        <div className="flex items-center gap-1.5">
          <StatusDot
            tone={source.error_count ? "error" : job ? "running" : "ok"}
            pulse={Boolean(job)}
          />
          <span className={cn("truncate text-[0.8125rem]", active && "font-medium")}>
            {shortPath(source.path)}
          </span>
        </div>
        <p className="mt-0.5 truncate pl-3.5 text-[0.6875rem] text-muted-foreground">
          {source.indexed_count}/{source.document_count} indexed · {relativeTime(source.last_scan_at)}
        </p>
      </button>

      {job ? (
        <div className="mt-1.5 pl-3.5">
          <Progress value={job.progress * 100} className="h-1" />
          <p className="mt-1 truncate text-[0.625rem] text-muted-foreground">
            {job.detail ?? job.label}
          </p>
        </div>
      ) : (
        <div className="mt-1 flex gap-0.5 pl-2.5 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
          <Button
            variant="ghost"
            size="sm"
            className="h-6 px-1.5 text-[0.6875rem]"
            onClick={() => onRescan(source.id)}
          >
            <RefreshCwIcon className="size-3" />
            Rescan
          </Button>

          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="ghost" size="sm" className="h-6 px-1.5 text-[0.6875rem]">
                <Trash2Icon className="size-3" />
                Remove
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Remove this source?</AlertDialogTitle>
                <AlertDialogDescription>
                  {source.document_count} document
                  {source.document_count === 1 ? "" : "s"} and their passages are deleted from the
                  index. The files on disk are not touched.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  onClick={() => onRemove(source.id)}
                  className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                >
                  Remove source
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
      )}
    </div>
  );
}
