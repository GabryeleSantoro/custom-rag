import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { DownloadIcon, HeartIcon, ScaleIcon } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { FitBadge } from "@/features/models/fit-badge";
import { bytes, count } from "@/lib/format";
import { api, type HubFile, type ModelRole } from "@/lib/ipc";
import { useJobs } from "@/lib/jobs-context";
import { keys } from "@/lib/queries";

const ROLE_LABEL: Record<ModelRole, string> = {
  embedding: "embedder",
  reranking: "reranker",
};

function modelJobId(repoId: string, filename: string) {
  return `${repoId}/${filename}`.replaceAll("/", "_").toLowerCase();
}

/**
 * Per-file download, with the embedder guard.
 *
 * Swapping the embedder invalidates every stored vector, so that one path asks
 * for a typed confirmation and states the cost before it starts.
 */
function DownloadButton({
  repoId,
  file,
  role,
  documentCount,
}: {
  repoId: string;
  file: HubFile;
  role: ModelRole;
  documentCount: number;
}) {
  const queryClient = useQueryClient();
  const { byModel } = useJobs();
  const [activate, setActivate] = useState(role !== "embedding");
  const [confirm, setConfirm] = useState("");

  const job = byModel.get(modelJobId(repoId, file.path));
  const needsReindexConfirm = role === "embedding" && activate;
  const confirmed = !needsReindexConfirm || confirm === "REINDEX";

  const download = useMutation({
    mutationFn: () =>
      api.download({
        repo_id: repoId,
        filename: file.path,
        role,
        activate,
        accept_reindex: needsReindexConfirm,
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: keys.models });
      toast.success("Download started", { description: file.path });
    },
    onError: (error: Error) => toast.error("Download failed", { description: error.message }),
  });

  if (job) {
    return (
      <div className="w-40">
        <Progress value={job.progress * 100} className="h-1.5" />
        <p className="mt-1 text-[0.625rem] text-muted-foreground">
          {job.detail ?? "Downloading"} · {Math.round(job.progress * 100)}%
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-end gap-1.5">
      <div className="flex items-center gap-2">
        <Switch
          id={`activate-${file.path}`}
          checked={activate}
          onCheckedChange={setActivate}
          aria-label="Use after download"
        />
        <Label htmlFor={`activate-${file.path}`} className="text-[0.6875rem] font-normal">
          Use it
        </Label>
        <Button
          size="sm"
          className="h-7"
          disabled={file.fit === "too-large" || download.isPending || !confirmed}
          onClick={() => download.mutate()}
        >
          <DownloadIcon className="size-3.5" />
          Download
        </Button>
      </div>

      {needsReindexConfirm ? (
        <div className="w-64 rounded-md border border-status-error/30 bg-status-error/8 p-2">
          <p className="text-[0.6875rem] leading-[1.45] text-status-error">
            Switching the embedder makes every stored vector meaningless. All {documentCount}{" "}
            document{documentCount === 1 ? "" : "s"} must be re-indexed before search works again.
          </p>
          <Input
            value={confirm}
            onChange={(event) => setConfirm(event.target.value)}
            placeholder="Type REINDEX to confirm"
            className="mt-1.5 h-7 text-[0.6875rem]"
          />
        </div>
      ) : null}
    </div>
  );
}

export function ModelDetailSheet({
  repoId,
  role,
  documentCount,
  onClose,
}: {
  repoId: string | null;
  role: ModelRole;
  documentCount: number;
  onClose: () => void;
}) {
  const detail = useQuery({
    queryKey: keys.hubDetail(repoId ?? ""),
    queryFn: () => api.hubDetail(repoId as string, role),
    enabled: Boolean(repoId),
  });

  return (
    <Sheet open={Boolean(repoId)} onOpenChange={(open) => !open && onClose()}>
      <SheetContent className="w-[34rem] sm:max-w-[34rem]">
        <SheetHeader>
          <SheetTitle className="font-mono text-sm">{repoId}</SheetTitle>
          <SheetDescription>
            Pick a quantisation. Smaller files lose some quality; the fit badge is measured against
            this machine, as the {ROLE_LABEL[role]}.
          </SheetDescription>
        </SheetHeader>

        <div className="flex-1 overflow-y-auto px-4 pb-4">
          {detail.isLoading ? (
            <div className="space-y-2">
              <Skeleton className="h-14 w-full" />
              <Skeleton className="h-14 w-full" />
            </div>
          ) : detail.isError ? (
            <p className="text-xs text-status-error">{(detail.error as Error).message}</p>
          ) : null}

          {detail.data ? (
            <>
              <div className="mb-4 flex flex-wrap gap-x-4 gap-y-1 text-[0.6875rem] text-muted-foreground">
                <span className="inline-flex items-center gap-1">
                  <DownloadIcon className="size-3" />
                  {count(detail.data.model.downloads)} downloads
                </span>
                <span className="inline-flex items-center gap-1">
                  <HeartIcon className="size-3" />
                  {count(detail.data.model.likes)}
                </span>
                {detail.data.model.license ? (
                  <span className="inline-flex items-center gap-1">
                    <ScaleIcon className="size-3" />
                    {detail.data.model.license}
                  </span>
                ) : null}
              </div>

              {detail.data.files.length === 0 ? (
                <p className="text-xs text-muted-foreground">
                  This repository publishes no GGUF files, so it cannot run in llama.cpp.
                </p>
              ) : null}

              <ul className="space-y-2">
                {detail.data.files.map((file) => (
                  <li
                    key={file.path}
                    className="flex items-start justify-between gap-3 rounded-lg border border-border bg-card p-3"
                  >
                    <div className="min-w-0">
                      <p className="truncate font-mono text-xs">{file.path}</p>
                      <div className="mt-1.5 flex items-center gap-2">
                        {file.quant ? (
                          <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[0.625rem]">
                            {file.quant}
                          </span>
                        ) : null}
                        <span className="font-mono text-[0.625rem] text-muted-foreground tabular-nums">
                          {bytes(file.size_bytes)}
                        </span>
                        <FitBadge fit={file.fit ?? "unknown"} note={file.fit_note} />
                      </div>
                    </div>

                    <DownloadButton
                      repoId={repoId as string}
                      file={file}
                      role={role}
                      documentCount={documentCount}
                    />
                  </li>
                ))}
              </ul>
            </>
          ) : null}
        </div>
      </SheetContent>
    </Sheet>
  );
}
