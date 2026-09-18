import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckIcon, CpuIcon, Trash2Icon } from "lucide-react";
import { toast } from "sonner";

import { ContextSidebar, SidebarSectionLabel } from "@/components/shell/context-sidebar";
import { Page, PageBody, PageHeader } from "@/components/shell/page";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { IconTooltip } from "@/components/ui/tooltip";
import { HubBrowser } from "@/features/models/hub-browser";
import { ModelDetailSheet } from "@/features/models/model-detail";
import { bytes, relativeTime } from "@/lib/format";
import { api, type InstalledModel, type ModelRole } from "@/lib/ipc";
import { healthQuery, hardwareQuery, keys, modelsQuery } from "@/lib/queries";
import { cn } from "@/lib/utils";

const ROLES: { role: ModelRole; label: string; blurb: string }[] = [
  {
    role: "embedding",
    label: "Embedder",
    blurb: "Turns passages into vectors. Changing it forces a full re-index.",
  },
  {
    role: "reranking",
    label: "Reranker",
    blurb: "Scores how well each candidate answers the question. Swappable at any time.",
  },
  {
    role: "generation",
    label: "Generation",
    blurb: "Writes the answer. Download one to run it in-app instead of connecting a server.",
  },
];

function InstalledCard({
  model,
  onActivate,
  onRemove,
}: {
  model: InstalledModel;
  onActivate: () => void;
  onRemove: () => void;
}) {
  return (
    <div
      className={cn(
        "flex items-start justify-between gap-4 rounded-lg border bg-card p-3",
        model.active ? "border-primary/40" : "border-border",
      )}
    >
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <p className="truncate text-[0.8125rem] font-medium">{model.name}</p>
          {model.active ? (
            <span className="inline-flex items-center gap-1 rounded-full bg-primary/12 px-1.5 py-0.5 text-[0.625rem] font-medium text-primary">
              <CheckIcon className="size-2.5" />
              Active
            </span>
          ) : null}
          {model.shipped ? (
            <span className="rounded-full bg-muted px-1.5 py-0.5 text-[0.625rem] text-muted-foreground">
              Shipped
            </span>
          ) : null}
        </div>
        <p className="truncate font-mono text-[0.625rem] text-muted-foreground">{model.repo_id}</p>
        <p className="mt-1.5 font-mono text-[0.625rem] text-muted-foreground tabular-nums">
          {model.quant ?? "—"} · {bytes(model.size_bytes)} · added {relativeTime(model.downloaded_at)}
        </p>
      </div>

      <div className="flex shrink-0 items-center gap-1">
        {!model.active ? (
          <Button variant="secondary" size="sm" className="h-7" onClick={onActivate}>
            Use
          </Button>
        ) : null}
        {!model.shipped && !model.active ? (
          <IconTooltip label="Remove">
            <Button
              variant="ghost"
              size="icon"
              className="size-7"
              aria-label="Remove"
              onClick={onRemove}
            >
              <Trash2Icon className="size-3.5" />
            </Button>
          </IconTooltip>
        ) : null}
      </div>
    </div>
  );
}

export function ModelsView() {
  const queryClient = useQueryClient();
  const [role, setRole] = useState<ModelRole>("reranking");
  const [detailRepo, setDetailRepo] = useState<string | null>(null);

  const models = useQuery(modelsQuery);
  const hardware = useQuery(hardwareQuery);
  const health = useQuery(healthQuery);

  const documentCount = health.data?.index.documents ?? 0;
  const forRole = (models.data?.installed ?? []).filter((model) => model.role === role);
  const installedRepos = new Set((models.data?.installed ?? []).map((model) => model.repo_id));

  const activate = useMutation({
    mutationFn: (model: InstalledModel) =>
      api.activateModel(model.id, model.role === "embedding"),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: keys.models });
      toast.success("Model activated");
    },
    onError: (error: Error) => toast.error("Could not activate", { description: error.message }),
  });

  const remove = useMutation({
    mutationFn: (id: string) => api.removeModel(id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: keys.models });
      toast.success("Model removed");
    },
    onError: (error: Error) => toast.error("Could not remove", { description: error.message }),
  });

  const meta = ROLES.find((entry) => entry.role === role)!;

  return (
    <>
      <ContextSidebar
        title="Models"
        footer={
          hardware.data ? (
            <div className="rounded-md bg-card p-2">
              <p className="flex items-center gap-1.5 text-[0.6875rem] font-medium">
                <CpuIcon className="size-3" />
                {hardware.data.gpu_name ?? hardware.data.gpu_backend.toUpperCase()}
              </p>
              <p className="mt-1 font-mono text-[0.625rem] text-muted-foreground tabular-nums">
                {Math.round(hardware.data.ram_mb / 1024)} GB RAM
                {hardware.data.vram_mb
                  ? ` · ${Math.round(hardware.data.vram_mb / 1024)} GB usable by the GPU`
                  : ""}
              </p>
              <p className="mt-0.5 text-[0.625rem] text-muted-foreground">
                Profile: {hardware.data.profile}
              </p>
            </div>
          ) : (
            <Skeleton className="h-14 w-full" />
          )
        }
      >
        <SidebarSectionLabel>Roles</SidebarSectionLabel>
        <nav className="flex flex-col gap-0.5">
          {ROLES.map((entry) => {
            const active = (models.data?.installed ?? []).find(
              (model) => model.role === entry.role && model.active,
            );
            return (
              <button
                key={entry.role}
                type="button"
                onClick={() => setRole(entry.role)}
                className={cn(
                  "rounded-md px-2 py-1.5 text-left transition-colors",
                  role === entry.role ? "bg-sidebar-accent" : "hover:bg-sidebar-accent/60",
                )}
              >
                <p
                  className={cn(
                    "text-[0.8125rem]",
                    role === entry.role ? "font-medium" : "text-sidebar-foreground",
                  )}
                >
                  {entry.label}
                </p>
                <p className="truncate text-[0.6875rem] text-muted-foreground">
                  {active ? active.name : "None active"}
                </p>
              </button>
            );
          })}
        </nav>
      </ContextSidebar>

      <Page>
        <PageHeader title={meta.label} description={meta.blurb} />

        <PageBody>
          <Tabs defaultValue="installed" className="h-full gap-0">
            <div className="border-b border-border px-5 pt-3">
              <TabsList>
                <TabsTrigger value="installed">Installed</TabsTrigger>
                <TabsTrigger value="hub">Browse Hugging Face</TabsTrigger>
              </TabsList>
            </div>

            <TabsContent value="installed" className="space-y-2 p-5">
              {models.isLoading ? <Skeleton className="h-20 w-full" /> : null}

              {forRole.length === 0 && !models.isLoading ? (
                <div className="rounded-lg border border-dashed border-border p-8 text-center">
                  <p className="text-sm font-medium">No {meta.label.toLowerCase()} installed</p>
                  <p className="mx-auto mt-1 max-w-sm text-xs text-muted-foreground">
                    {role === "generation"
                      ? "Download one here to answer questions without an external server, or connect a provider in Settings."
                      : "Browse Hugging Face to install one."}
                  </p>
                </div>
              ) : null}

              {forRole.map((model) => (
                <InstalledCard
                  key={model.id}
                  model={model}
                  onActivate={() => activate.mutate(model)}
                  onRemove={() => remove.mutate(model.id)}
                />
              ))}
            </TabsContent>

            <TabsContent value="hub" className="p-0">
              <HubBrowser role={role} installedRepos={installedRepos} onOpen={setDetailRepo} />
            </TabsContent>
          </Tabs>
        </PageBody>
      </Page>

      <ModelDetailSheet
        repoId={detailRepo}
        role={role}
        documentCount={documentCount}
        onClose={() => setDetailRepo(null)}
      />
    </>
  );
}
