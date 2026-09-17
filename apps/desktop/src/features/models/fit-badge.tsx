import { CpuIcon, MemoryStickIcon, XIcon, ZapIcon } from "lucide-react";

import type { FitVerdict } from "@/lib/ipc";
import { cn } from "@/lib/utils";

const FIT: Record<FitVerdict, { label: string; icon: typeof ZapIcon; className: string }> = {
  vram: {
    label: "Fits in VRAM",
    icon: ZapIcon,
    className: "border-status-ok/30 bg-status-ok/10 text-status-ok",
  },
  ram: {
    label: "Fits in RAM",
    icon: MemoryStickIcon,
    className: "border-status-warn/30 bg-status-warn/10 text-status-warn",
  },
  "too-large": {
    label: "Too large",
    icon: XIcon,
    className: "border-status-error/30 bg-status-error/10 text-status-error",
  },
  unknown: {
    label: "Unknown",
    icon: CpuIcon,
    className: "border-border bg-muted text-muted-foreground",
  },
};

/** What the hardware probe says about running this file on this machine. */
export function FitBadge({
  fit,
  note,
  className,
}: {
  fit: FitVerdict;
  note?: string | null;
  className?: string;
}) {
  const meta = FIT[fit] ?? FIT.unknown;
  return (
    <span
      title={note ?? undefined}
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-1.5 py-0.5 text-[0.625rem] font-medium whitespace-nowrap",
        meta.className,
        className,
      )}
    >
      <meta.icon className="size-2.5" />
      {meta.label}
    </span>
  );
}
