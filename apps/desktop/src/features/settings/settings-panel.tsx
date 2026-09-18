import { useEffect, useState } from "react";
import { useParams } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CloudIcon, HardDriveIcon, PlusIcon, Trash2Icon } from "lucide-react";
import { toast } from "sonner";

import { Page, PageBody, PageHeader } from "@/components/shell/page";
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
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Skeleton } from "@/components/ui/skeleton";
import { IconTooltip } from "@/components/ui/tooltip";
import { ConnectionDialog } from "@/features/settings/connection-dialog";
import { SETTINGS_SECTIONS } from "@/features/settings/settings-sidebar";
import {
  api,
  shell,
  type AppSettings,
  type PerformanceSettings,
  type RetrievalSettings,
} from "@/lib/ipc";
import { connectionsQuery, hardwareQuery, keys, settingsQuery } from "@/lib/queries";
import { cn } from "@/lib/utils";

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="grid gap-1.5 sm:grid-cols-[14rem_1fr] sm:items-baseline sm:gap-6">
      <div>
        <Label className="text-[0.8125rem]">{label}</Label>
        {hint ? <p className="mt-0.5 text-[0.6875rem] text-muted-foreground">{hint}</p> : null}
      </div>
      <div className="max-w-xs">{children}</div>
    </div>
  );
}

function useSettingsDraft() {
  const queryClient = useQueryClient();
  const settings = useQuery(settingsQuery);
  const [draft, setDraft] = useState<AppSettings | null>(null);

  useEffect(() => {
    if (settings.data) setDraft(settings.data);
  }, [settings.data]);

  const save = useMutation({
    mutationFn: (patch: Parameters<typeof api.patchSettings>[0]) => api.patchSettings(patch),
    onSuccess: (updated) => {
      queryClient.setQueryData(keys.settings, updated);
      toast.success("Settings saved");
    },
    onError: (error: Error) => toast.error("Could not save", { description: error.message }),
  });

  return { settings, draft, setDraft, save };
}

function ConnectionsSection() {
  const queryClient = useQueryClient();
  const connections = useQuery(connectionsQuery);

  const activate = useMutation({
    mutationFn: (id: string) => api.activateConnection(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: keys.connections }),
  });

  const remove = useMutation({
    mutationFn: async (id: string) => {
      await api.deleteConnection(id);
      await shell.keychainDelete(id).catch(() => undefined);
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: keys.connections });
      toast.success("Connection removed");
    },
  });

  return (
    <div className="space-y-3">
      {connections.isLoading ? <Skeleton className="h-20 w-full" /> : null}

      {(connections.data ?? []).map((connection) => (
        <div
          key={connection.id}
          className={cn(
            "flex items-start justify-between gap-4 rounded-lg border bg-card p-3",
            connection.active ? "border-primary/40" : "border-border",
          )}
        >
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <p className="truncate text-[0.8125rem] font-medium">{connection.name}</p>
              {connection.active ? (
                <span className="rounded-full bg-primary/12 px-1.5 py-0.5 text-[0.625rem] font-medium text-primary">
                  Active
                </span>
              ) : null}
              {connection.is_remote ? (
                <span className="inline-flex items-center gap-1 rounded-full bg-status-warn/12 px-1.5 py-0.5 text-[0.625rem] font-medium text-status-warn">
                  <CloudIcon className="size-2.5" />
                  Remote
                </span>
              ) : null}
            </div>
            <p className="truncate font-mono text-[0.625rem] text-muted-foreground">
              {connection.model_id}
              {connection.base_url ? ` · ${connection.base_url}` : ""}
            </p>
            <p className="mt-1 font-mono text-[0.625rem] text-muted-foreground tabular-nums">
              {connection.context_window.toLocaleString()} ctx ·{" "}
              {connection.max_output_tokens.toLocaleString()} out
              {connection.has_api_key ? " · key in keychain" : ""}
            </p>
          </div>

          <div className="flex shrink-0 items-center gap-1">
            {!connection.active ? (
              <Button
                variant="secondary"
                size="sm"
                className="h-7"
                onClick={() => activate.mutate(connection.id)}
              >
                Use
              </Button>
            ) : null}
            <ConnectionDialog
              connection={connection}
              trigger={
                <Button variant="ghost" size="sm" className="h-7">
                  Edit
                </Button>
              }
            />
            <IconTooltip label="Remove">
              <Button
                variant="ghost"
                size="icon"
                className="size-7"
                aria-label="Remove"
                onClick={() => remove.mutate(connection.id)}
              >
                <Trash2Icon className="size-3.5" />
              </Button>
            </IconTooltip>
          </div>
        </div>
      ))}

      <ConnectionDialog
        trigger={
          <Button variant="secondary" size="sm">
            <PlusIcon className="size-4" />
            Add connection
          </Button>
        }
      />
    </div>
  );
}

function RetrievalSection() {
  const { draft, setDraft, save } = useSettingsDraft();
  if (!draft) return <Skeleton className="h-40 w-full" />;

  const retrieval = draft.retrieval as RetrievalSettings;
  const set = (key: keyof RetrievalSettings, value: number) =>
    setDraft({ ...draft, retrieval: { ...retrieval, [key]: value } });

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <Field
          label="Passages sent to the model"
          hint="Top-k after the reranker. More context, slower answers."
        >
          <Input
            type="number"
            value={retrieval.top_k}
            onChange={(event) => set("top_k", Number(event.target.value) || 1)}
          />
        </Field>

        <Field
          label="Minimum score"
          hint="Below this, a passage is dropped. This is what lets the app answer “not found”."
        >
          <Input
            type="number"
            step="0.05"
            min="0"
            max="1"
            value={retrieval.min_score}
            onChange={(event) => set("min_score", Number(event.target.value))}
          />
        </Field>

        <Field
          label="Rerank candidates"
          hint="Cost is linear here. Forty on a GPU, twenty on CPU."
        >
          <Input
            type="number"
            value={retrieval.rerank_candidates}
            onChange={(event) => set("rerank_candidates", Number(event.target.value) || 1)}
          />
        </Field>
      </div>

      <Collapsible>
        <CollapsibleTrigger className="text-[0.8125rem] font-medium text-muted-foreground hover:text-foreground">
          Advanced
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-4 space-y-4">
          <Field label="Dense candidates" hint="Vector search depth before fusion.">
            <Input
              type="number"
              value={retrieval.dense_top_k}
              onChange={(event) => set("dense_top_k", Number(event.target.value) || 1)}
            />
          </Field>
          <Field label="Keyword candidates" hint="BM25 depth before fusion.">
            <Input
              type="number"
              value={retrieval.bm25_top_k}
              onChange={(event) => set("bm25_top_k", Number(event.target.value) || 1)}
            />
          </Field>
          <Field
            label="RRF constant"
            hint="Small values trust rank one heavily; large values reward broad agreement."
          >
            <Input
              type="number"
              value={retrieval.rrf_k}
              onChange={(event) => set("rrf_k", Number(event.target.value) || 1)}
            />
          </Field>
          <Field label="Context budget" hint="Tokens of passages packed into the prompt.">
            <Input
              type="number"
              value={retrieval.context_token_budget}
              onChange={(event) => set("context_token_budget", Number(event.target.value) || 1)}
            />
          </Field>
          <Field label="History budget" hint="Tokens of chat history before older turns are summarised.">
            <Input
              type="number"
              value={retrieval.history_token_budget}
              onChange={(event) => set("history_token_budget", Number(event.target.value) || 1)}
            />
          </Field>
        </CollapsibleContent>
      </Collapsible>

      <Button
        disabled={save.isPending}
        onClick={() => save.mutate({ retrieval: draft.retrieval })}
      >
        Save retrieval settings
      </Button>
    </div>
  );
}

function PerformanceSection() {
  const { draft, setDraft, save } = useSettingsDraft();
  const hardware = useQuery(hardwareQuery);
  if (!draft) return <Skeleton className="h-40 w-full" />;

  const performance = draft.performance as PerformanceSettings;
  const set = <K extends keyof PerformanceSettings>(key: K, value: PerformanceSettings[K]) =>
    setDraft({ ...draft, performance: { ...performance, [key]: value } });

  return (
    <div className="space-y-6">
      {hardware.data ? (
        <div className="rounded-lg border border-border bg-card p-3 text-[0.6875rem]">
          <p className="font-medium">
            Detected: {hardware.data.gpu_name ?? hardware.data.gpu_backend.toUpperCase()} ·{" "}
            {Math.round(hardware.data.ram_mb / 1024)} GB RAM · {hardware.data.cpu_count} cores
          </p>
          <p className="mt-0.5 text-muted-foreground">
            Suggested profile: {hardware.data.profile}. Overriding it is fine; it only changes
            defaults.
          </p>
        </div>
      ) : null}

      <Field label="Profile" hint="Sets candidate counts and batch sizes.">
        <Select
          value={performance.profile}
          onValueChange={(value) => set("profile", value as PerformanceSettings["profile"])}
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="cpu">CPU only</SelectItem>
            <SelectItem value="balanced">Balanced (GPU ≤ 8 GB)</SelectItem>
            <SelectItem value="gpu">GPU (&gt; 8 GB)</SelectItem>
          </SelectContent>
        </Select>
      </Field>

      <Field label="Embedding batch" hint="Passages embedded per call during indexing.">
        <Input
          type="number"
          value={performance.embed_batch}
          onChange={(event) => set("embed_batch", Number(event.target.value) || 1)}
        />
      </Field>

      <Field label="GPU layers" hint="Layers offloaded to the GPU. 999 means all of them.">
        <Input
          type="number"
          value={performance.gpu_layers}
          onChange={(event) => set("gpu_layers", Number(event.target.value) || 0)}
        />
      </Field>

      <Field label="Parallel parsers" hint="Files parsed at once. Raise it on many cores.">
        <Input
          type="number"
          value={performance.max_parallel_parsers}
          onChange={(event) => set("max_parallel_parsers", Number(event.target.value) || 1)}
        />
      </Field>

      <Button
        disabled={save.isPending}
        onClick={() => save.mutate({ performance: draft.performance })}
      >
        Save performance settings
      </Button>
    </div>
  );
}

function StorageSection() {
  const queryClient = useQueryClient();
  const { draft, setDraft, save } = useSettingsDraft();

  const wipe = useMutation({
    mutationFn: (keepConnections: boolean) => api.wipe(keepConnections),
    onSuccess: () => {
      void queryClient.invalidateQueries();
      toast.success("Index wiped");
    },
    onError: (error: Error) => toast.error("Wipe failed", { description: error.message }),
  });

  if (!draft) return <Skeleton className="h-40 w-full" />;

  return (
    <div className="space-y-6">
      <Field label="Storage location" hint="Where the index, models and settings live.">
        <Input
          value={draft.storage_path}
          onChange={(event) => setDraft({ ...draft, storage_path: event.target.value })}
        />
      </Field>

      <Field label="Crash reports" hint="Opt-in, and never includes document content.">
        <div className="flex items-center gap-2">
          <Switch
            checked={draft.telemetry}
            onCheckedChange={(checked) => setDraft({ ...draft, telemetry: checked })}
          />
          <span className="text-[0.8125rem]">{draft.telemetry ? "Enabled" : "Disabled"}</span>
        </div>
      </Field>

      <Button
        disabled={save.isPending}
        onClick={() =>
          save.mutate({ storage_path: draft.storage_path, telemetry: draft.telemetry })
        }
      >
        Save storage settings
      </Button>

      <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-4">
        <h3 className="flex items-center gap-1.5 text-[0.8125rem] font-semibold text-destructive">
          <HardDriveIcon className="size-3.5" />
          Wipe all data
        </h3>
        <p className="mt-1 max-w-prose text-[0.6875rem] leading-[1.5] text-muted-foreground">
          Deletes every source, document, passage and chat from the index. Files on disk are not
          touched, and downloaded models are kept.
        </p>

        <AlertDialog>
          <AlertDialogTrigger asChild>
            <Button variant="destructive" size="sm" className="mt-3">
              Wipe index
            </Button>
          </AlertDialogTrigger>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Wipe the index?</AlertDialogTitle>
              <AlertDialogDescription>
                Every indexed document and every chat is removed and you will be taken back
                through setup. Your files and your connections stay.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction
                onClick={() => wipe.mutate(true)}
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              >
                Wipe everything
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
    </div>
  );
}

export function SettingsPanel() {
  const { section } = useParams({ from: "/_shell/settings/$section" });
  const meta = SETTINGS_SECTIONS.find((entry) => entry.slug === section);

  return (
    <Page>
      <PageHeader title={meta?.label ?? "Settings"} description={meta?.blurb} />
      <PageBody>
        <div className="max-w-3xl p-5">
          {section === "connections" ? <ConnectionsSection /> : null}
          {section === "retrieval" ? <RetrievalSection /> : null}
          {section === "performance" ? <PerformanceSection /> : null}
          {section === "storage" ? <StorageSection /> : null}
        </div>
      </PageBody>
    </Page>
  );
}
