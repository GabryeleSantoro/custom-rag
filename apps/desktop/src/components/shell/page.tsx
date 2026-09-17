import { trafficLightGutter } from "@/lib/platform";
import { cn } from "@/lib/utils";

/**
 * Main-pane frame. The header doubles as the window drag strip on macOS,
 * where there is no native title bar to grab.
 */
export function Page({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("flex min-w-0 flex-1 flex-col", className)}>{children}</div>;
}

export function PageHeader({
  title,
  description,
  actions,
  className,
  children,
}: {
  title?: React.ReactNode;
  description?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
  children?: React.ReactNode;
}) {
  return (
    <header
      className={cn(
        "drag-region flex shrink-0 items-center gap-3 border-b border-border px-5 pb-3",
        className,
      )}
      style={{ paddingTop: (trafficLightGutter ? 0 : 0) + 12 }}
    >
      {children ?? (
        <div className="min-w-0 flex-1">
          {title ? (
            <h1 className="truncate text-sm font-semibold tracking-tight">{title}</h1>
          ) : null}
          {description ? (
            <p className="truncate text-xs text-muted-foreground">{description}</p>
          ) : null}
        </div>
      )}
      {actions ? <div className="no-drag flex items-center gap-2">{actions}</div> : null}
    </header>
  );
}

export function PageBody({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return <div className={cn("min-h-0 flex-1 overflow-auto", className)}>{children}</div>;
}
