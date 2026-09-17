import { cn } from "@/lib/utils";

type Tone = "idle" | "running" | "ok" | "warn" | "error";

const TONE: Record<Tone, string> = {
  idle: "bg-status-idle",
  running: "bg-status-running",
  ok: "bg-status-ok",
  warn: "bg-status-warn",
  error: "bg-status-error",
};

const TEXT: Record<Tone, string> = {
  idle: "text-muted-foreground",
  running: "text-status-running",
  ok: "text-status-ok",
  warn: "text-status-warn",
  error: "text-status-error",
};

export function StatusDot({
  tone,
  pulse,
  className,
}: {
  tone: Tone;
  pulse?: boolean;
  className?: string;
}) {
  return (
    <span className={cn("relative inline-flex size-2 shrink-0", className)}>
      {pulse ? (
        <span className={cn("absolute inline-flex size-full animate-ping rounded-full opacity-60", TONE[tone])} />
      ) : null}
      <span className={cn("relative inline-flex size-2 rounded-full", TONE[tone])} />
    </span>
  );
}

export function StatusChip({
  tone,
  label,
  pulse,
  className,
}: {
  tone: Tone;
  label: string;
  pulse?: boolean;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border border-border px-2 py-0.5",
        "text-[0.6875rem] font-medium whitespace-nowrap",
        TEXT[tone],
        className,
      )}
    >
      <StatusDot tone={tone} pulse={pulse} />
      {label}
    </span>
  );
}

/** Pipeline stage to tone. Shared by the Library table and the job chips. */
export function documentTone(status: string): Tone {
  switch (status) {
    case "indexed":
      return "ok";
    case "error":
      return "error";
    case "skipped":
      return "warn";
    case "queued":
      return "idle";
    default:
      return "running";
  }
}
