import { useEffect, useMemo, useRef, useState } from "react";
import { SearchIcon } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";

type Level = "error" | "warn" | "info";

// Python logs arrive on stderr like everything else, so the stream says nothing:
// the level comes from the text.
const ERROR = /\b(ERROR|CRITICAL)\b|Traceback|\bfailed\b|\bbroken\b|panicked/;
const WARN = /\bWARNING\b/;

export function levelOf(line: string): Level {
  if (ERROR.test(line)) return "error";
  if (WARN.test(line)) return "warn";
  return "info";
}

/** `fill` stretches the pane to its parent's height instead of a fixed box. */
export function LogPane({ lines, fill = false }: { lines: string[]; fill?: boolean }) {
  const { t } = useTranslation();
  const [filter, setFilter] = useState("");
  const [problemsOnly, setProblemsOnly] = useState(false);
  const [follow, setFollow] = useState(true);
  const viewport = useRef<HTMLDivElement>(null);

  const shown = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    return lines.filter(
      (line) =>
        (!problemsOnly || levelOf(line) !== "info") &&
        (!needle || line.toLowerCase().includes(needle)),
    );
  }, [lines, filter, problemsOnly]);

  useEffect(() => {
    if (!follow) return;
    const node = viewport.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [shown, follow]);

  return (
    <div
      className={cn(
        "flex flex-col overflow-hidden rounded-lg border border-border bg-card",
        fill && "min-h-0 flex-1",
      )}
    >
      <div className="flex items-center gap-3 border-b border-border px-2.5 py-2">
        <div className="relative min-w-0 flex-1">
          <SearchIcon className="pointer-events-none absolute top-1/2 left-2 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
            placeholder={t("logs.filter")}
            className="h-7 pl-7 text-xs"
          />
        </div>
        <label className="flex shrink-0 items-center gap-2 text-[0.6875rem] text-muted-foreground">
          {t("logs.problemsOnly")}
          <Switch checked={problemsOnly} onCheckedChange={setProblemsOnly} />
        </label>
        <label className="flex shrink-0 items-center gap-2 text-[0.6875rem] text-muted-foreground">
          {t("logs.follow")}
          <Switch checked={follow} onCheckedChange={setFollow} />
        </label>
        <span className="shrink-0 font-mono text-[0.625rem] text-muted-foreground tabular-nums">
          {shown.length}/{lines.length}
        </span>
      </div>

      <div
        ref={viewport}
        className={cn("overflow-auto bg-background/40 px-2.5 py-2", fill ? "min-h-0 flex-1" : "h-72")}
      >
        {shown.length === 0 ? (
          <p className="py-8 text-center text-xs text-muted-foreground">
            {lines.length === 0 ? t("logs.empty") : t("logs.noMatch")}
          </p>
        ) : (
          shown.map((line, index) => {
            const level = levelOf(line);
            return (
              <p
                key={`${index}-${line.slice(0, 32)}`}
                className={cn(
                  "selectable font-mono text-[0.6875rem] leading-[1.55] break-words whitespace-pre-wrap",
                  level === "error"
                    ? "text-status-error"
                    : level === "warn"
                      ? "text-status-warn"
                      : "text-muted-foreground",
                )}
              >
                {line}
              </p>
            );
          })
        )}
      </div>
    </div>
  );
}
