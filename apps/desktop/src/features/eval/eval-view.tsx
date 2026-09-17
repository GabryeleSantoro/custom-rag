import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { FlaskConicalIcon, PlayIcon, SquareIcon } from "lucide-react";

import { ContextSidebar, SidebarSectionLabel } from "@/components/shell/context-sidebar";
import { Page, PageBody, PageHeader } from "@/components/shell/page";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ms } from "@/lib/format";
import {
  streamEval,
  type EvalResult,
  type EvalQuestionResult,
  type StreamHandle,
} from "@/lib/ipc";
import { evalSetsQuery, healthQuery } from "@/lib/queries";
import { cn } from "@/lib/utils";

type RunState =
  | { phase: "idle" }
  | { phase: "running"; completed: number; total: number; question: string }
  | { phase: "done"; result: EvalResult }
  | { phase: "failed"; message: string };

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/** Signed delta against the stored baseline, coloured by direction. */
function Delta({ value }: { value: number }) {
  const flat = Math.abs(value) < 0.0005;
  return (
    <span
      className={cn(
        "font-mono text-xs tabular-nums",
        flat ? "text-muted-foreground" : value > 0 ? "text-status-ok" : "text-status-error",
      )}
    >
      {flat ? "±0.0 pt" : `${value > 0 ? "+" : "−"}${Math.abs(value * 100).toFixed(1)} pt`}
    </span>
  );
}

function MetricsTable({ result }: { result: EvalResult }) {
  const metrics = result.metrics;
  const baseline = metrics.baseline_recall_at_6;

  const rows: { label: string; hint: string; value: string; baseline?: number; raw?: number }[] = [
    {
      label: "Recall@1",
      hint: "The right document was the first passage returned.",
      value: percent(metrics.recall_at_1),
    },
    {
      label: "Recall@6",
      hint: "The right document appeared anywhere in the packed context.",
      value: percent(metrics.recall_at_6),
      baseline: baseline ?? undefined,
      raw: metrics.recall_at_6,
    },
    {
      label: "MRR",
      hint: "1/rank of the first correct passage, averaged. Rewards ranking, not just presence.",
      value: metrics.mrr.toFixed(3),
    },
    {
      label: "nDCG@6",
      hint: "Gain discounted by position across the whole returned list.",
      value: metrics.ndcg_at_6.toFixed(3),
    },
  ];

  return (
    <div className="overflow-hidden rounded-lg border border-border bg-card">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Metric</TableHead>
            <TableHead className="w-24 text-right">Value</TableHead>
            <TableHead className="w-28 text-right">Baseline</TableHead>
            <TableHead className="w-28 text-right">Δ</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row) => (
            <TableRow key={row.label}>
              <TableCell>
                <p className="text-[0.8125rem] font-medium">{row.label}</p>
                <p className="text-[0.625rem] leading-[1.45] text-muted-foreground">{row.hint}</p>
              </TableCell>
              <TableCell className="text-right font-mono text-xs tabular-nums">
                {row.value}
              </TableCell>
              <TableCell className="text-right font-mono text-xs text-muted-foreground tabular-nums">
                {row.baseline != null ? percent(row.baseline) : "—"}
              </TableCell>
              <TableCell className="text-right">
                {row.baseline != null && row.raw != null ? (
                  <Delta value={row.raw - row.baseline} />
                ) : (
                  <span className="font-mono text-xs text-muted-foreground">—</span>
                )}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>

      <div className="flex flex-wrap gap-x-6 gap-y-1 border-t border-border px-4 py-2.5">
        <span className="font-mono text-[0.6875rem] text-muted-foreground tabular-nums">
          p50 {ms(metrics.latency_p50.total_ms)}
        </span>
        <span className="font-mono text-[0.6875rem] text-muted-foreground tabular-nums">
          p95 {ms(metrics.latency_p95.total_ms)}
        </span>
        <span className="font-mono text-[0.6875rem] text-muted-foreground tabular-nums">
          {metrics.n_questions} questions · set “{metrics.set_name}”
        </span>
      </div>
    </div>
  );
}

function QuestionsTable({ questions }: { questions: EvalQuestionResult[] }) {
  const misses = questions.filter((question) => !question.hit).length;

  return (
    <div className="overflow-hidden rounded-lg border border-border bg-card">
      <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
        <h3 className="text-[0.8125rem] font-semibold tracking-tight">Per question</h3>
        <span className="text-[0.6875rem] text-muted-foreground">
          {misses === 0 ? "No misses" : `${misses} missed`}
        </span>
      </div>
      <div className="max-h-96 overflow-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-10 text-right">#</TableHead>
              <TableHead>Question</TableHead>
              <TableHead className="w-16 text-right">Rank</TableHead>
              <TableHead className="w-20 text-right">nDCG</TableHead>
              <TableHead className="w-20 text-right">Latency</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {questions.map((question, index) => (
              <TableRow key={`${index}-${question.q}`} className={question.hit ? "" : "bg-status-error/5"}>
                <TableCell className="text-right font-mono text-[0.6875rem] text-muted-foreground tabular-nums">
                  {index + 1}
                </TableCell>
                <TableCell className="selectable max-w-0 truncate text-xs">{question.q}</TableCell>
                <TableCell
                  className={cn(
                    "text-right font-mono text-xs tabular-nums",
                    question.hit ? "" : "text-status-error",
                  )}
                >
                  {question.rank ?? "miss"}
                </TableCell>
                <TableCell className="text-right font-mono text-xs tabular-nums">
                  {question.ndcg.toFixed(3)}
                </TableCell>
                <TableCell className="text-right font-mono text-xs text-muted-foreground tabular-nums">
                  {ms(question.latency_ms)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}

export function EvalView() {
  const health = useQuery(healthQuery);
  const sets = useQuery(evalSetsQuery);
  const [setName, setSetName] = useState<string | null>(null);
  const [compareBaseline, setCompareBaseline] = useState(true);
  const [run, setRun] = useState<RunState>({ phase: "idle" });
  const handle = useRef<StreamHandle | null>(null);

  const list = sets.data ?? [];
  const selected = setName ?? list[0]?.name ?? null;
  const devMode = health.data?.dev_mode ?? false;

  useEffect(() => () => void handle.current?.cancel(), []);

  function start() {
    if (!selected) return;
    setRun({ phase: "running", completed: 0, total: 0, question: "" });

    handle.current = streamEval(
      { set_name: selected, compare_baseline: compareBaseline },
      {
        onEvent: (event) => {
          if (event.event === "progress") {
            setRun({
              phase: "running",
              completed: event.data.completed,
              total: event.data.total,
              question: event.data.question,
            });
          } else {
            setRun({ phase: "done", result: event.data });
          }
        },
        onFailed: (message) => setRun({ phase: "failed", message }),
        onClosed: () =>
          setRun((current) =>
            current.phase === "running"
              ? { phase: "failed", message: "The run ended before any metrics arrived." }
              : current,
          ),
      },
    );
  }

  function stop() {
    void handle.current?.cancel();
    setRun({ phase: "idle" });
  }

  const running = run.phase === "running";

  return (
    <>
      <ContextSidebar title="Golden sets">
        <SidebarSectionLabel>Sets</SidebarSectionLabel>
        {sets.isLoading ? <Skeleton className="mx-2 h-16" /> : null}
        {list.map((set) => (
          <button
            key={set.name}
            type="button"
            disabled={running}
            onClick={() => setSetName(set.name)}
            className={cn(
              "w-full rounded-md px-2 py-1.5 text-left",
              set.name === selected
                ? "bg-sidebar-accent"
                : "hover:bg-sidebar-accent/60 disabled:hover:bg-transparent",
            )}
          >
            <p className="text-[0.8125rem] font-medium">{set.name}</p>
            <p className="text-[0.6875rem] leading-[1.4] text-muted-foreground">
              {set.n_questions} questions
              {set.description ? ` · ${set.description}` : ""}
            </p>
          </button>
        ))}

        <SidebarSectionLabel>Options</SidebarSectionLabel>
        <label className="flex items-center justify-between gap-2 rounded-md px-2 py-1.5">
          <span className="text-[0.8125rem]">Diff against baseline</span>
          <Switch
            checked={compareBaseline}
            onCheckedChange={setCompareBaseline}
            disabled={running}
          />
        </label>
      </ContextSidebar>

      <Page>
        <PageHeader
          title="Eval runner"
          description="Retrieval quality on a labelled set, measured the same way every time"
          actions={
            running ? (
              <Button variant="secondary" size="sm" className="h-7" onClick={stop}>
                <SquareIcon className="size-3.5" />
                Stop
              </Button>
            ) : (
              <Button size="sm" className="h-7" disabled={!devMode || !selected} onClick={start}>
                <PlayIcon className="size-3.5" />
                Run
              </Button>
            )
          }
        />

        <PageBody>
          <div className="mx-auto max-w-3xl space-y-4 p-5">
            {!devMode ? (
              <div className="rounded-lg border border-dashed border-border p-8 text-center">
                <FlaskConicalIcon
                  className="mx-auto size-6 text-muted-foreground"
                  strokeWidth={1.5}
                />
                <p className="mt-2 text-sm font-medium">Dev builds only</p>
                <p className="mx-auto mt-1 max-w-sm text-xs text-muted-foreground">
                  The eval route is disabled in release builds: it replays a labelled set through
                  the live retriever, which is not something a shipped app should do on demand.
                </p>
              </div>
            ) : null}

            {running ? (
              <div className="rounded-lg border border-border bg-card p-4">
                <div className="flex items-baseline justify-between gap-4">
                  <p className="text-[0.8125rem] font-medium">Running “{selected}”</p>
                  <span className="font-mono text-[0.6875rem] text-muted-foreground tabular-nums">
                    {run.completed}/{run.total || "…"}
                  </span>
                </div>
                <Progress
                  className="mt-2 h-1"
                  value={run.total ? Math.round((run.completed / run.total) * 100) : 0}
                />
                <p className="mt-2 truncate font-mono text-[0.625rem] text-muted-foreground">
                  {run.question || "starting…"}
                </p>
              </div>
            ) : null}

            {run.phase === "failed" ? (
              <div className="rounded-lg border border-status-error/30 bg-status-error/8 px-3 py-2.5 text-xs text-status-error">
                {run.message}
              </div>
            ) : null}

            {run.phase === "done" ? (
              <>
                <MetricsTable result={run.result} />
                <QuestionsTable questions={run.result.questions} />
              </>
            ) : null}

            {devMode && run.phase === "idle" ? (
              <div className="rounded-lg border border-dashed border-border p-8 text-center">
                <FlaskConicalIcon
                  className="mx-auto size-6 text-muted-foreground"
                  strokeWidth={1.5}
                />
                <p className="mt-2 text-sm font-medium">Nothing measured yet</p>
                <p className="mx-auto mt-1 max-w-sm text-xs text-muted-foreground">
                  A run replays every question in the selected set through the live retriever and
                  reports where the right document landed. Change a retrieval setting, run again,
                  and read the delta.
                </p>
              </div>
            ) : null}
          </div>
        </PageBody>
      </Page>
    </>
  );
}
