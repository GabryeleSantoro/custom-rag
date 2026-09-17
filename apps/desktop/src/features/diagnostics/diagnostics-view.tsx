import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ClipboardCheckIcon,
  ClipboardCopyIcon,
  RotateCwIcon,
  SearchIcon,
} from "lucide-react";
import { toast } from "sonner";

import { Page, PageBody, PageHeader } from "@/components/shell/page";
import { StatusChip } from "@/components/status";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { bytes, count, duration, relativeTime } from "@/lib/format";
import { shell, type Health, type HardwareInfo, type ProcessStatus } from "@/lib/ipc";
import { hardwareQuery, healthQuery, keys, logsQuery, sidecarsQuery } from "@/lib/queries";
import { cn } from "@/lib/utils";

type ProcState = ProcessStatus["state"];

const STATE_TONE = {
  stopped: "idle",
  starting: "running",
  ready: "ok",
  restarting: "warn",
  error: "error",
} as const;

/** One row per managed process, whoever reported it. */
type ProcRow = {
  name: string;
  role: string;
  state: ProcState;
  pid: number | null;
  port: number | null;
  uptime_s: number;
  restarts: number;
  detail: string | null;
  /** True when the Rust shell owns the process lifecycle. */
  managed: boolean;
};

const ROLE_BLURB: Record<string, string> = {
  ragcore: "Python core: ingestion, retrieval, the API every screen reads from.",
  embedding: "Embedding server. Turns chunks and questions into vectors.",
  reranking: "Cross-encoder. Reorders candidates before the prompt is packed.",
  generation: "In-app chat model. Only running when a local model is active.",
};

function Section({
  title,
  description,
  action,
  children,
}: {
  title: string;
  description?: string;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="space-y-2.5">
      <div className="flex items-end justify-between gap-4">
        <div>
          <h2 className="text-[0.8125rem] font-semibold tracking-tight">{title}</h2>
          {description ? (
            <p className="text-[0.6875rem] text-muted-foreground">{description}</p>
          ) : null}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}

function Stat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="min-w-0">
      <p className="text-[0.625rem] tracking-wide text-muted-foreground uppercase">{label}</p>
      <p className="selectable truncate font-mono text-xs tabular-nums">{value}</p>
    </div>
  );
}

function ProcessCard({ proc }: { proc: ProcRow }) {
  return (
    <div
      className={cn(
        "rounded-lg border bg-card p-3",
        proc.state === "error" ? "border-status-error/35" : "border-border",
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate text-[0.8125rem] font-medium">{proc.name}</p>
          <p className="mt-0.5 text-[0.6875rem] leading-[1.45] text-muted-foreground">
            {ROLE_BLURB[proc.role] ?? proc.role}
          </p>
        </div>
        <StatusChip
          tone={STATE_TONE[proc.state]}
          label={proc.state}
          pulse={proc.state === "starting" || proc.state === "restarting"}
        />
      </div>

      <div className="mt-3 grid grid-cols-4 gap-3 border-t border-border pt-2.5">
        <Stat label="pid" value={proc.pid ?? "—"} />
        <Stat label="port" value={proc.port ?? "—"} />
        <Stat label="uptime" value={duration(proc.uptime_s)} />
        <Stat label="restarts" value={proc.restarts} />
      </div>

      {proc.detail ? (
        <p
          className={cn(
            "selectable mt-2 font-mono text-[0.625rem] leading-[1.5] break-words",
            proc.state === "error" ? "text-status-error" : "text-muted-foreground",
          )}
        >
          {proc.detail}
        </p>
      ) : null}

      {!proc.managed ? (
        <p className="mt-2 text-[0.625rem] text-muted-foreground">
          Reported by the core, not supervised by the shell.
        </p>
      ) : null}
    </div>
  );
}

function IndexSection({ health }: { health: Health | undefined }) {
  if (!health) return <Skeleton className="h-24 w-full" />;
  const index = health.index;

  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        <Stat label="documents" value={count(index.documents)} />
        <Stat label="chunks" value={count(index.chunks)} />
        <Stat label="parents" value={count(index.parents)} />
        <Stat label="topics" value={count(index.topics)} />
        <Stat label="on disk" value={bytes(index.size_bytes)} />
      </div>
      <div className="mt-3 grid grid-cols-2 gap-3 border-t border-border pt-2.5 sm:grid-cols-4">
        <Stat label="embedder" value={index.embed_model} />
        <Stat label="dim" value={index.embed_dim} />
        <Stat label="reranker" value={index.reranker_model} />
        <Stat label="last indexed" value={relativeTime(index.last_indexed_at)} />
      </div>
      <p className="mt-2.5 text-[0.6875rem] text-muted-foreground">
        Index schema v{index.schema_version}. Changing the embedder rewrites every vector, so the
        schema version moves with it.
      </p>
    </div>
  );
}

function EnvironmentSection({
  health,
  hardware,
}: {
  health: Health | undefined;
  hardware: HardwareInfo | undefined;
}) {
  if (!hardware) return <Skeleton className="h-20 w-full" />;

  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="platform" value={`${hardware.os} ${hardware.arch}`} />
        <Stat label="cpu threads" value={hardware.cpu_count} />
        <Stat label="ram" value={`${(hardware.ram_mb / 1024).toFixed(1)} GB`} />
        <Stat
          label="gpu"
          value={
            hardware.gpu_backend === "cpu"
              ? "none"
              : `${hardware.gpu_backend} · ${(hardware.vram_mb / 1024).toFixed(1)} GB`
          }
        />
      </div>
      <div className="mt-3 grid grid-cols-2 gap-3 border-t border-border pt-2.5 sm:grid-cols-4">
        <Stat label="profile" value={hardware.profile} />
        <Stat label="core version" value={health?.version ?? "—"} />
        <Stat label="core uptime" value={duration(health?.uptime_s)} />
        <Stat label="build" value={health?.dev_mode ? "dev" : "release"} />
      </div>
      {health?.stub ? (
        <p className="mt-2.5 text-[0.6875rem] text-status-warn">
          The core is serving fixture data. Retrieval and answers are shaped like the real thing
          but are not computed over your documents.
        </p>
      ) : null}
    </div>
  );
}

function LogPane({ lines }: { lines: string[] }) {
  const [filter, setFilter] = useState("");
  const [follow, setFollow] = useState(true);
  const viewport = useRef<HTMLDivElement>(null);

  const shown = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    return needle ? lines.filter((line) => line.toLowerCase().includes(needle)) : lines;
  }, [lines, filter]);

  useEffect(() => {
    if (!follow) return;
    const node = viewport.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [shown, follow]);

  return (
    <div className="overflow-hidden rounded-lg border border-border bg-card">
      <div className="flex items-center gap-3 border-b border-border px-2.5 py-2">
        <div className="relative min-w-0 flex-1">
          <SearchIcon className="pointer-events-none absolute top-1/2 left-2 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
            placeholder="Filter lines"
            className="h-7 pl-7 text-xs"
          />
        </div>
        <label className="flex shrink-0 items-center gap-2 text-[0.6875rem] text-muted-foreground">
          Follow
          <Switch checked={follow} onCheckedChange={setFollow} />
        </label>
        <span className="shrink-0 font-mono text-[0.625rem] text-muted-foreground tabular-nums">
          {shown.length}/{lines.length}
        </span>
      </div>

      <div ref={viewport} className="h-72 overflow-auto bg-background/40 px-2.5 py-2">
        {shown.length === 0 ? (
          <p className="py-8 text-center text-xs text-muted-foreground">
            {lines.length === 0 ? "No output yet." : "No line matches that filter."}
          </p>
        ) : (
          shown.map((line, index) => (
            <p
              key={`${index}-${line.slice(0, 24)}`}
              className={cn(
                "selectable font-mono text-[0.6875rem] leading-[1.55] break-words whitespace-pre-wrap",
                line.startsWith("[err]") ? "text-status-error" : "text-muted-foreground",
              )}
            >
              {line}
            </p>
          ))
        )}
      </div>
    </div>
  );
}

/**
 * The text behind "copy debug report". Deliberately excludes the session token
 * and every API key: this string is written to be pasted into a bug report.
 */
function debugReport(input: {
  health: Health | undefined;
  hardware: HardwareInfo | undefined;
  processes: ProcRow[];
  logs: string[];
}): string {
  const { health, hardware, processes, logs } = input;
  const lines: string[] = ["# custom-rag debug report", `generated: ${new Date().toISOString()}`, ""];

  lines.push("## environment");
  if (hardware) {
    lines.push(`- platform: ${hardware.os} ${hardware.arch}, ${hardware.cpu_count} threads`);
    lines.push(`- ram: ${hardware.ram_mb} MB`);
    lines.push(`- gpu: ${hardware.gpu_backend}${hardware.gpu_name ? ` (${hardware.gpu_name})` : ""}, ${hardware.vram_mb} MB VRAM`);
    lines.push(`- profile: ${hardware.profile}`);
  } else {
    lines.push("- unavailable");
  }
  lines.push(
    `- core: ${health?.version ?? "?"} (${health?.dev_mode ? "dev" : "release"}${health?.stub ? ", stub data" : ""})`,
    `- status: ${health?.status ?? "unreachable"}`,
    "",
  );

  lines.push("## processes");
  for (const proc of processes) {
    lines.push(
      `- ${proc.name} [${proc.role}] ${proc.state} pid=${proc.pid ?? "-"} port=${proc.port ?? "-"} ` +
        `uptime=${Math.round(proc.uptime_s)}s restarts=${proc.restarts}` +
        (proc.detail ? ` detail="${proc.detail}"` : ""),
    );
  }
  lines.push("");

  if (health) {
    const index = health.index;
    lines.push(
      "## index",
      `- documents: ${index.documents}, chunks: ${index.chunks}, parents: ${index.parents}`,
      `- size: ${index.size_bytes} bytes, schema v${index.schema_version}`,
      `- embedder: ${index.embed_model} (dim ${index.embed_dim})`,
      `- reranker: ${index.reranker_model}`,
      `- last indexed: ${index.last_indexed_at ?? "never"}`,
      "",
    );
  }

  lines.push("## last log lines", "```", ...logs.slice(-120), "```");
  return lines.join("\n");
}

export function DiagnosticsView() {
  const queryClient = useQueryClient();
  const health = useQuery(healthQuery);
  const hardware = useQuery(hardwareQuery);
  const sidecars = useQuery(sidecarsQuery);
  const logs = useQuery(logsQuery);
  const [copied, setCopied] = useState(false);

  const processes = useMemo<ProcRow[]>(() => {
    const managed: ProcRow[] = (sidecars.data ?? []).map((proc) => ({ ...proc, managed: true }));
    const covered = new Set(managed.map((proc) => proc.role));
    const reported: ProcRow[] = (health.data?.processes ?? [])
      .filter((proc) => !covered.has(proc.role))
      .map((proc) => ({
        name: proc.name,
        role: proc.role,
        state: proc.state,
        pid: proc.pid ?? null,
        port: proc.port ?? null,
        uptime_s: proc.uptime_s,
        restarts: proc.restarts,
        detail: proc.detail ?? null,
        managed: false,
      }));
    return [...managed, ...reported];
  }, [sidecars.data, health.data]);

  const lines = logs.data ?? [];

  async function copyReport() {
    const text = debugReport({
      health: health.data,
      hardware: hardware.data,
      processes,
      logs: lines,
    });
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      toast.success("Debug report copied", {
        description: "No API keys or session tokens are included.",
      });
    } catch {
      toast.error("Could not reach the clipboard");
    }
  }

  async function restart() {
    try {
      await shell.restart();
      toast.success("Restarting ragcore", { description: "The supervisor will bring it back." });
      void queryClient.invalidateQueries({ queryKey: keys.sidecars });
      void queryClient.invalidateQueries({ queryKey: keys.health });
    } catch (error) {
      toast.error("Restart failed", { description: String(error) });
    }
  }

  const unreachable = health.isError;

  return (
    <Page>
      <PageHeader
        title="Diagnostics"
        description="What is running, what it indexed, and what it printed"
        actions={
          <>
            <Button variant="ghost" size="sm" className="h-7" onClick={() => void restart()}>
              <RotateCwIcon className="size-3.5" />
              Restart core
            </Button>
            <Button variant="secondary" size="sm" className="h-7" onClick={() => void copyReport()}>
              {copied ? (
                <ClipboardCheckIcon className="size-3.5" />
              ) : (
                <ClipboardCopyIcon className="size-3.5" />
              )}
              Copy debug report
            </Button>
          </>
        }
      />

      <PageBody>
        {unreachable ? (
          <div className="border-b border-status-error/25 bg-status-error/8 px-5 py-2 text-xs text-status-error">
            The core is not answering on loopback. The process cards below still show what the
            shell knows, and the log tail is the fastest way to see why.
          </div>
        ) : null}

        <div className="mx-auto max-w-4xl space-y-6 p-5">
          <Section
            title="Processes"
            description="Everything this app started on your machine."
          >
            {sidecars.isLoading && processes.length === 0 ? (
              <Skeleton className="h-28 w-full" />
            ) : (
              <div className="grid gap-2.5 lg:grid-cols-2">
                {processes.map((proc) => (
                  <ProcessCard key={`${proc.role}-${proc.name}`} proc={proc} />
                ))}
              </div>
            )}
          </Section>

          <Section title="Index" description="What retrieval currently searches over.">
            <IndexSection health={health.data} />
          </Section>

          <Section title="Environment" description="Detected hardware and the build in use.">
            <EnvironmentSection health={health.data} hardware={hardware.data} />
          </Section>

          <Section
            title="Core log"
            description="Last 500 lines of the sidecar's stdout and stderr."
          >
            <LogPane lines={lines} />
          </Section>
        </div>
      </PageBody>
    </Page>
  );
}
