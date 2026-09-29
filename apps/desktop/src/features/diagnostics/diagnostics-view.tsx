import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ClipboardCheckIcon, ClipboardCopyIcon, RotateCwIcon } from "lucide-react";
import { toast } from "sonner";

import { Page, PageBody, PageHeader } from "@/components/shell/page";
import { StatusChip } from "@/components/status";
import { Button } from "@/components/ui/button";
import { LogPane } from "@/components/log-pane";
import { Skeleton } from "@/components/ui/skeleton";
import { errorText } from "@/lib/errors";
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

const ROLE_BLURB_KEY: Record<string, string> = {
  ragcore: "diagnostics.roleRagcore",
  embedding: "diagnostics.roleEmbedding",
  reranking: "diagnostics.roleReranking",
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
  const { t } = useTranslation();
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
            {ROLE_BLURB_KEY[proc.role] ? t(ROLE_BLURB_KEY[proc.role]) : proc.role}
          </p>
        </div>
        <StatusChip
          tone={STATE_TONE[proc.state]}
          label={t(`diagnostics.state.${proc.state}`)}
          pulse={proc.state === "starting" || proc.state === "restarting"}
        />
      </div>

      <div className="mt-3 grid grid-cols-4 gap-3 border-t border-border pt-2.5">
        <Stat label={t("diagnostics.pid")} value={proc.pid ?? "—"} />
        <Stat label={t("diagnostics.port")} value={proc.port ?? "—"} />
        <Stat label={t("diagnostics.uptime")} value={duration(proc.uptime_s)} />
        <Stat label={t("diagnostics.restarts")} value={proc.restarts} />
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
          {t("diagnostics.unsupervised")}
        </p>
      ) : null}
    </div>
  );
}

function IndexSection({ health }: { health: Health | undefined }) {
  const { t } = useTranslation();
  if (!health) return <Skeleton className="h-24 w-full" />;
  const index = health.index;

  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        <Stat label={t("diagnostics.documents")} value={count(index.documents)} />
        <Stat label={t("diagnostics.chunks")} value={count(index.chunks)} />
        <Stat label={t("diagnostics.parents")} value={count(index.parents)} />
        <Stat label={t("diagnostics.topics")} value={count(index.topics)} />
        <Stat label={t("diagnostics.onDisk")} value={bytes(index.size_bytes)} />
      </div>
      <div className="mt-3 grid grid-cols-2 gap-3 border-t border-border pt-2.5 sm:grid-cols-4">
        <Stat label={t("diagnostics.embedder")} value={index.embed_model} />
        <Stat label={t("diagnostics.dim")} value={index.embed_dim} />
        <Stat label={t("diagnostics.reranker")} value={index.reranker_model} />
        <Stat label={t("diagnostics.lastIndexed")} value={relativeTime(index.last_indexed_at)} />
      </div>
      <p className="mt-2.5 text-[0.6875rem] text-muted-foreground">
        {t("diagnostics.schema", { version: index.schema_version })}
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
  const { t } = useTranslation();
  if (!hardware) return <Skeleton className="h-20 w-full" />;

  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label={t("diagnostics.platform")} value={`${hardware.os} ${hardware.arch}`} />
        <Stat label={t("diagnostics.threads")} value={hardware.cpu_count} />
        <Stat label={t("diagnostics.ram")} value={`${(hardware.ram_mb / 1024).toFixed(1)} GB`} />
        <Stat
          label={t("diagnostics.gpu")}
          value={
            hardware.gpu_backend === "cpu"
              ? t("diagnostics.none")
              : `${hardware.gpu_backend} · ${(hardware.vram_mb / 1024).toFixed(1)} GB`
          }
        />
      </div>
      <div className="mt-3 grid grid-cols-2 gap-3 border-t border-border pt-2.5 sm:grid-cols-4">
        <Stat label={t("diagnostics.profile")} value={t(`onboarding.hardware.profileName.${hardware.profile}`)} />
        <Stat label={t("diagnostics.coreVersion")} value={health?.version ?? "—"} />
        <Stat label={t("diagnostics.coreUptime")} value={duration(health?.uptime_s)} />
        <Stat label={t("diagnostics.build")} value={health?.dev_mode ? t("diagnostics.dev") : t("diagnostics.release")} />
      </div>
      {health?.stub ? (
        <p className="mt-2.5 text-[0.6875rem] text-status-warn">
          {t("diagnostics.stub")}
        </p>
      ) : null}
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
  const { t } = useTranslation();
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
      toast.success(t("diagnostics.reportCopied"), {
        description: t("diagnostics.reportNoSecrets"),
      });
    } catch {
      toast.error(t("diagnostics.clipboardFailed"));
    }
  }

  async function restart() {
    try {
      await shell.restart();
      toast.success(t("diagnostics.restarting"), { description: t("diagnostics.restartingHint") });
      void queryClient.invalidateQueries({ queryKey: keys.sidecars });
      void queryClient.invalidateQueries({ queryKey: keys.health });
    } catch (error) {
      toast.error(t("diagnostics.restartFailed"), { description: errorText(error) });
    }
  }

  const unreachable = health.isError;

  return (
    <Page>
      <PageHeader
        title={t("nav.diagnostics")}
        description={t("diagnostics.lead")}
        actions={
          <>
            <Button variant="ghost" size="sm" className="h-7" onClick={() => void restart()}>
              <RotateCwIcon className="size-3.5" />
              {t("diagnostics.restart")}
            </Button>
            <Button variant="secondary" size="sm" className="h-7" onClick={() => void copyReport()}>
              {copied ? (
                <ClipboardCheckIcon className="size-3.5" />
              ) : (
                <ClipboardCopyIcon className="size-3.5" />
              )}
              {t("diagnostics.copyReport")}
            </Button>
          </>
        }
      />

      <PageBody>
        {unreachable ? (
          <div className="border-b border-status-error/25 bg-status-error/8 px-5 py-2 text-xs text-status-error">
            {t("diagnostics.unreachable")}
          </div>
        ) : null}

        <div className="mx-auto max-w-4xl space-y-6 p-5">
          <Section
            title={t("diagnostics.processes")}
            description={t("diagnostics.processesHint")}
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

          <Section title={t("diagnostics.index")} description={t("diagnostics.indexHint")}>
            <IndexSection health={health.data} />
          </Section>

          <Section title={t("diagnostics.environment")} description={t("diagnostics.environmentHint")}>
            <EnvironmentSection health={health.data} hardware={hardware.data} />
          </Section>

          <Section
            title={t("diagnostics.recentLog")}
            description={t("diagnostics.recentLogHint")}
          >
            <LogPane lines={lines} />
          </Section>
        </div>
      </PageBody>
    </Page>
  );
}
