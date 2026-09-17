import { useMemo, useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowRightIcon,
  BookOpenIcon,
  CheckIcon,
  CloudIcon,
  CpuIcon,
  DownloadIcon,
  FolderPlusIcon,
  Loader2Icon,
  LockIcon,
  QuoteIcon,
  ServerIcon,
} from "lucide-react";
import { toast } from "sonner";

import { StatusChip } from "@/components/status";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { AddSourceDialog } from "@/features/library/add-source-dialog";
import { ConnectionDialog } from "@/features/settings/connection-dialog";
import { bytes } from "@/lib/format";
import { useJobs } from "@/lib/jobs-context";
import { api, type HardwareInfo, type InstalledModel, type ModelRole } from "@/lib/ipc";
import {
  connectionsQuery,
  hardwareQuery,
  healthQuery,
  keys,
  modelsQuery,
  sourcesQuery,
} from "@/lib/queries";
import { cn } from "@/lib/utils";

const STEPS = [
  { id: "welcome", label: "Welcome" },
  { id: "hardware", label: "Your machine" },
  { id: "models", label: "Core models" },
  { id: "connection", label: "Answer model" },
  { id: "library", label: "First folder" },
  { id: "done", label: "Ready" },
] as const;

type StepId = (typeof STEPS)[number]["id"];

/**
 * The two models the pipeline cannot run without. Retrieval is entirely local,
 * so these download once and then never phone home.
 */
const CORE_MODELS: {
  role: ModelRole;
  name: string;
  repo_id: string;
  filename: string;
  why: string;
}[] = [
  {
    role: "embedding",
    name: "Qwen3 Embedding 0.6B",
    repo_id: "Qwen/Qwen3-Embedding-0.6B-GGUF",
    filename: "Qwen3-Embedding-0.6B-Q8_0.gguf",
    why: "Turns every chunk and every question into a vector. Swapping it later means re-indexing.",
  },
  {
    role: "reranking",
    name: "Qwen3 Reranker 0.6B",
    repo_id: "Qwen/Qwen3-Reranker-0.6B-GGUF",
    filename: "Qwen3-Reranker-0.6B-Q8_0.gguf",
    why: "Reads the question and each candidate together, then reorders them. This is what keeps answers on topic.",
  },
];

const PROFILE_NOTE: Record<HardwareInfo["profile"], string> = {
  gpu: "Plenty of VRAM. Models run on the GPU, batches are large, parsing runs wide.",
  balanced: "Enough memory to keep the retrieval models resident while you work.",
  cpu: "No usable GPU found. Everything runs on CPU with small batches — slower, still correct.",
};

function StepFrame({
  title,
  lead,
  children,
}: {
  title: string;
  lead?: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="space-y-5">
      <div className="space-y-1.5">
        <h1 className="text-lg font-semibold tracking-tight">{title}</h1>
        {lead ? (
          <p className="max-w-prose text-[0.8125rem] leading-[1.55] text-muted-foreground">
            {lead}
          </p>
        ) : null}
      </div>
      {children}
    </div>
  );
}

function Row({
  icon: Icon,
  title,
  body,
  aside,
  tone,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: React.ReactNode;
  body?: React.ReactNode;
  aside?: React.ReactNode;
  tone?: "ok";
}) {
  return (
    <div
      className={cn(
        "flex items-start gap-3 rounded-lg border bg-card p-3",
        tone === "ok" ? "border-status-ok/30" : "border-border",
      )}
    >
      <Icon className={cn("mt-0.5 size-4 shrink-0", tone === "ok" ? "text-status-ok" : "text-muted-foreground")} />
      <div className="min-w-0 flex-1">
        <p className="text-[0.8125rem] font-medium">{title}</p>
        {body ? (
          <div className="mt-0.5 text-[0.6875rem] leading-[1.5] text-muted-foreground">{body}</div>
        ) : null}
      </div>
      {aside ? <div className="shrink-0">{aside}</div> : null}
    </div>
  );
}

function WelcomeStep() {
  return (
    <StepFrame
      title="A librarian for your own files"
      lead="Point it at folders you already have. It reads them, indexes them locally, and answers questions with the passage it used sitting next to the answer."
    >
      <div className="space-y-2">
        <Row
          icon={LockIcon}
          title="Your documents stay on this machine"
          body="Parsing, embedding and search all run here. Only the question and the retrieved passages ever reach a model you choose — and that model can be a local one."
        />
        <Row
          icon={QuoteIcon}
          title="Every claim points at a page"
          body="Answers carry citations. Click one and the reader opens on the exact span it came from, so a wrong answer is visibly wrong."
        />
        <Row
          icon={BookOpenIcon}
          title="Four short steps"
          body="Check your hardware, get the two retrieval models, pick who writes the answers, add a folder."
        />
      </div>
    </StepFrame>
  );
}

function HardwareStep({ hardware }: { hardware: HardwareInfo | undefined }) {
  if (!hardware) {
    return (
      <StepFrame title="Looking at your machine" lead="Reading CPU, memory and GPU.">
        <Skeleton className="h-24 w-full" />
      </StepFrame>
    );
  }

  const gpu =
    hardware.gpu_backend === "cpu"
      ? "No GPU backend detected"
      : `${hardware.gpu_backend}${hardware.gpu_name ? ` · ${hardware.gpu_name}` : ""} · ${(
          hardware.vram_mb / 1024
        ).toFixed(1)} GB VRAM`;

  return (
    <StepFrame
      title="Your machine"
      lead="Defaults come from what is actually here. Everything below is changeable later in Settings → Performance."
    >
      <div className="grid gap-2 sm:grid-cols-2">
        <Row
          icon={CpuIcon}
          title={`${hardware.os} · ${hardware.arch}`}
          body={`${hardware.cpu_count} threads · ${(hardware.ram_mb / 1024).toFixed(1)} GB RAM`}
        />
        <Row icon={ServerIcon} title="Graphics" body={gpu} />
      </div>

      <div className="rounded-lg border border-primary/35 bg-primary/5 p-3">
        <p className="text-[0.8125rem] font-medium">
          Profile: <span className="capitalize">{hardware.profile}</span>
        </p>
        <p className="mt-0.5 text-[0.6875rem] leading-[1.5] text-muted-foreground">
          {PROFILE_NOTE[hardware.profile]}
        </p>
      </div>
    </StepFrame>
  );
}

function ModelRow({
  spec,
  installed,
}: {
  spec: (typeof CORE_MODELS)[number];
  installed: InstalledModel | undefined;
}) {
  const queryClient = useQueryClient();
  const { byModel } = useJobs();
  const [startedModelId, setStartedModelId] = useState<string | null>(null);

  const job = startedModelId ? byModel.get(startedModelId) : undefined;

  const download = useMutation({
    mutationFn: () =>
      api.download({
        repo_id: spec.repo_id,
        filename: spec.filename,
        role: spec.role,
        activate: true,
        accept_reindex: true,
      }),
    onSuccess: (started) => {
      setStartedModelId(started.model_id ?? null);
      void queryClient.invalidateQueries({ queryKey: keys.jobs });
    },
    onError: (error: Error) =>
      toast.error(`Could not start the ${spec.role} download`, { description: error.message }),
  });

  if (installed) {
    return (
      <Row
        icon={CheckIcon}
        tone="ok"
        title={installed.name}
        body={
          <>
            {spec.why}
            <span className="mt-1 block font-mono text-[0.625rem]">
              {bytes(installed.size_bytes)}
              {installed.quant ? ` · ${installed.quant}` : ""}
              {installed.sha256 ? ` · sha256 ${installed.sha256.slice(0, 12)}…` : ""}
            </span>
          </>
        }
        aside={<StatusChip tone="ok" label="verified" />}
      />
    );
  }

  return (
    <Row
      icon={DownloadIcon}
      title={spec.name}
      body={
        <>
          {spec.why}
          {job ? (
            <span className="mt-2 block space-y-1">
              <Progress value={Math.round(job.progress * 100)} className="h-1" />
              <span className="block font-mono text-[0.625rem]">
                {job.detail ?? job.label} · {Math.round(job.progress * 100)}%
              </span>
            </span>
          ) : null}
        </>
      }
      aside={
        <Button
          size="sm"
          className="h-7"
          disabled={download.isPending || (job != null && job.state === "running")}
          onClick={() => download.mutate()}
        >
          {download.isPending || job?.state === "running" ? (
            <Loader2Icon className="size-3.5 animate-spin" />
          ) : (
            <DownloadIcon className="size-3.5" />
          )}
          Download
        </Button>
      }
    />
  );
}

function ModelsStep({ installed }: { installed: InstalledModel[] }) {
  return (
    <StepFrame
      title="The two models retrieval needs"
      lead="Both run locally and are small enough to stay resident. Downloads resume where they stopped and are checked against their sha256 before being used."
    >
      <div className="space-y-2">
        {CORE_MODELS.map((spec) => (
          <ModelRow
            key={spec.role}
            spec={spec}
            installed={installed.find((model) => model.role === spec.role && model.active)}
          />
        ))}
      </div>
    </StepFrame>
  );
}

function ConnectionStep() {
  const connections = useQuery(connectionsQuery);
  const list = connections.data ?? [];

  return (
    <StepFrame
      title="Who writes the answers"
      lead="Retrieval is local either way. This only decides which model turns the retrieved passages into prose."
    >
      {connections.isLoading ? <Skeleton className="h-20 w-full" /> : null}

      <div className="space-y-2">
        {list.map((connection) => (
          <Row
            key={connection.id}
            icon={connection.is_remote ? CloudIcon : ServerIcon}
            tone={connection.active ? "ok" : undefined}
            title={
              <span className="flex items-center gap-2">
                {connection.name}
                {connection.active ? (
                  <span className="rounded-full bg-primary/12 px-1.5 py-0.5 text-[0.625rem] font-medium text-primary">
                    Active
                  </span>
                ) : null}
              </span>
            }
            body={
              <span className="font-mono text-[0.625rem]">
                {connection.model_id}
                {connection.base_url ? ` · ${connection.base_url}` : ""}
              </span>
            }
            aside={
              <ConnectionDialog
                connection={connection}
                trigger={
                  <Button variant="ghost" size="sm" className="h-7">
                    Edit
                  </Button>
                }
              />
            }
          />
        ))}
      </div>

      <ConnectionDialog
        trigger={
          <Button variant={list.length === 0 ? "default" : "secondary"} size="sm" className="h-7">
            {list.length === 0 ? "Add a connection" : "Add another"}
          </Button>
        }
      />

      {list.some((connection) => connection.is_remote && connection.active) ? (
        <p className="rounded-md border border-status-warn/30 bg-status-warn/8 px-2.5 py-2 text-[0.6875rem] text-status-warn">
          The active connection is remote. Your question and the retrieved passages leave this
          machine when you ask something. Chat shows a banner whenever that is the case.
        </p>
      ) : null}
    </StepFrame>
  );
}

function LibraryStep() {
  const sources = useQuery(sourcesQuery);
  const { bySource } = useJobs();
  const list = sources.data ?? [];

  return (
    <StepFrame
      title="Point it at a folder"
      lead="Files are read in place — nothing is copied or moved. Start with one folder; add the rest from the Library whenever you like."
    >
      {sources.isLoading ? <Skeleton className="h-20 w-full" /> : null}

      <div className="space-y-2">
        {list.map((source) => {
          const job = bySource.get(source.id);
          return (
            <Row
              key={source.id}
              icon={job ? Loader2Icon : CheckIcon}
              tone={job ? undefined : "ok"}
              title={source.path}
              body={
                job ? (
                  <>
                    <span className="mt-1 block">
                      <Progress value={Math.round(job.progress * 100)} className="h-1" />
                    </span>
                    <span className="mt-1 block">{job.detail ?? job.label}</span>
                  </>
                ) : (
                  `${source.indexed_count} of ${source.document_count} documents indexed` +
                  (source.error_count ? ` · ${source.error_count} failed` : "")
                )
              }
            />
          );
        })}
      </div>

      <AddSourceDialog
        trigger={
          <Button variant={list.length === 0 ? "default" : "secondary"} size="sm" className="h-7">
            <FolderPlusIcon className="size-3.5" />
            {list.length === 0 ? "Choose a folder" : "Add another folder"}
          </Button>
        }
      />
    </StepFrame>
  );
}

function DoneStep({ documents }: { documents: number }) {
  return (
    <StepFrame
      title="Set up"
      lead="Ask something in Chat. If an answer has no citation it says so rather than making one up — that is the whole point."
    >
      <div className="space-y-2">
        <Row
          icon={CheckIcon}
          tone="ok"
          title={`${documents} document${documents === 1 ? "" : "s"} in the index`}
          body="The Library shows every file, its status, and why anything failed."
        />
        <Row
          icon={QuoteIcon}
          title="Click a citation"
          body="It opens the Reader on the page the passage came from, with the span highlighted."
        />
        <Row
          icon={CpuIcon}
          title="Diagnostics knows what is running"
          body="Process cards, the core's log tail and a debug report you can paste into an issue."
        />
      </div>
    </StepFrame>
  );
}

export function OnboardingView() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [index, setIndex] = useState(0);

  const hardware = useQuery(hardwareQuery);
  const health = useQuery(healthQuery);
  const models = useQuery(modelsQuery);
  const connections = useQuery(connectionsQuery);
  const sources = useQuery(sourcesQuery);

  const installed = useMemo(() => models.data?.installed ?? [], [models.data]);
  const step: StepId = STEPS[index].id;

  const finish = useMutation({
    mutationFn: () => api.patchSettings({ onboarded: true }),
    onSuccess: (settings) => {
      queryClient.setQueryData(keys.settings, settings);
      void navigate({ to: "/chat", replace: true });
    },
    onError: (error: Error) =>
      toast.error("Could not save the setup", { description: error.message }),
  });

  const coreReady = CORE_MODELS.every((spec) =>
    installed.some((model) => model.role === spec.role && model.active),
  );

  const blocked =
    (step === "hardware" && !hardware.data) || (step === "models" && !coreReady);

  const optional =
    (step === "connection" && (connections.data ?? []).length === 0) ||
    (step === "library" && (sources.data ?? []).length === 0);

  function next() {
    if (step === "done") {
      finish.mutate();
      return;
    }
    setIndex((current) => Math.min(current + 1, STEPS.length - 1));
  }

  return (
    <div className="flex h-full overflow-hidden">
      <aside className="flex w-56 shrink-0 flex-col border-r border-sidebar-border bg-sidebar">
        <div className="drag-region h-10 shrink-0" />
        <nav className="space-y-0.5 px-2">
          {STEPS.map((entry, position) => {
            const state = position < index ? "past" : position === index ? "current" : "future";
            return (
              <button
                key={entry.id}
                type="button"
                disabled={state === "future"}
                onClick={() => setIndex(position)}
                className={cn(
                  "flex w-full items-center gap-2.5 rounded-md px-2 py-1.5 text-left text-[0.8125rem]",
                  state === "current" && "bg-sidebar-accent font-medium text-sidebar-foreground",
                  state === "past" && "text-muted-foreground hover:bg-sidebar-accent/60",
                  state === "future" && "text-muted-foreground/50",
                )}
              >
                <span
                  className={cn(
                    "grid size-4 shrink-0 place-items-center rounded-full border text-[0.5625rem] tabular-nums",
                    state === "past"
                      ? "border-status-ok/50 text-status-ok"
                      : state === "current"
                        ? "border-primary text-primary"
                        : "border-border",
                  )}
                >
                  {state === "past" ? <CheckIcon className="size-2.5" /> : position + 1}
                </span>
                {entry.label}
              </button>
            );
          })}
        </nav>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <div className="drag-region h-10 shrink-0" />

        <div className="min-h-0 flex-1 overflow-auto">
          <div className="mx-auto max-w-2xl px-8 pb-10">
            {step === "welcome" ? <WelcomeStep /> : null}
            {step === "hardware" ? <HardwareStep hardware={hardware.data} /> : null}
            {step === "models" ? <ModelsStep installed={installed} /> : null}
            {step === "connection" ? <ConnectionStep /> : null}
            {step === "library" ? <LibraryStep /> : null}
            {step === "done" ? <DoneStep documents={health.data?.index.documents ?? 0} /> : null}
          </div>
        </div>

        <footer className="flex shrink-0 items-center justify-between gap-3 border-t border-border px-8 py-3">
          <Button
            variant="ghost"
            size="sm"
            className="h-8"
            disabled={index === 0}
            onClick={() => setIndex((current) => Math.max(current - 1, 0))}
          >
            Back
          </Button>

          <div className="flex items-center gap-2">
            {optional ? (
              <span className="text-[0.6875rem] text-muted-foreground">
                You can do this later.
              </span>
            ) : null}
            <Button size="sm" className="h-8" disabled={blocked || finish.isPending} onClick={next}>
              {finish.isPending ? <Loader2Icon className="size-3.5 animate-spin" /> : null}
              {step === "done" ? "Open chat" : optional ? "Skip for now" : "Continue"}
              {step === "done" ? null : <ArrowRightIcon className="size-3.5" />}
            </Button>
          </div>
        </footer>
      </div>
    </div>
  );
}
