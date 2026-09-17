import { Link, useRouterState } from "@tanstack/react-router";
import {
  ActivityIcon,
  BoxesIcon,
  FlaskConicalIcon,
  LibraryIcon,
  MessagesSquareIcon,
  SettingsIcon,
  type LucideIcon,
} from "lucide-react";

import { ThemeToggle } from "@/components/shell/theme-provider";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { trafficLightGutter } from "@/lib/platform";
import { cn } from "@/lib/utils";

type RailItem = {
  to: string;
  label: string;
  icon: LucideIcon;
  /** Matches nested routes, e.g. /reader/* belongs to Library. */
  match: string[];
  devOnly?: boolean;
};

const PRIMARY: RailItem[] = [
  { to: "/chat", label: "Chat", icon: MessagesSquareIcon, match: ["/chat", "/"] },
  { to: "/library", label: "Library", icon: LibraryIcon, match: ["/library", "/reader"] },
  { to: "/models", label: "Models", icon: BoxesIcon, match: ["/models"] },
  { to: "/settings", label: "Settings", icon: SettingsIcon, match: ["/settings"] },
];

const SECONDARY: RailItem[] = [
  { to: "/diagnostics", label: "Diagnostics", icon: ActivityIcon, match: ["/diagnostics"] },
  { to: "/eval", label: "Eval runner", icon: FlaskConicalIcon, match: ["/eval"], devOnly: true },
];

function RailButton({ item, active }: { item: RailItem; active: boolean }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Link
          to={item.to}
          aria-label={item.label}
          aria-current={active ? "page" : undefined}
          className={cn(
            "no-drag relative grid size-9 place-items-center rounded-md transition-colors",
            "text-rail-foreground hover:bg-sidebar-accent hover:text-foreground",
            active && "bg-sidebar-accent text-foreground",
          )}
        >
          {/* Active marker rides the rail edge so the icon itself stays unstyled. */}
          <span
            className={cn(
              "absolute -left-2 h-5 w-0.75 rounded-full bg-primary transition-opacity",
              active ? "opacity-100" : "opacity-0",
            )}
          />
          <item.icon className="size-4.5" strokeWidth={1.75} />
        </Link>
      </TooltipTrigger>
      <TooltipContent side="right">{item.label}</TooltipContent>
    </Tooltip>
  );
}

export function IconRail({ devMode = false }: { devMode?: boolean }) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const isActive = (item: RailItem) =>
    item.match.some((m) => (m === "/" ? pathname === "/" : pathname.startsWith(m)));

  return (
    <nav
      className="drag-region flex w-rail shrink-0 flex-col items-center gap-1 border-r border-sidebar-border bg-rail pb-3"
      style={{ paddingTop: trafficLightGutter + 12 }}
    >
      {PRIMARY.map((item) => (
        <RailButton key={item.to} item={item} active={isActive(item)} />
      ))}

      <div className="flex-1" />

      {SECONDARY.filter((item) => !item.devOnly || devMode).map((item) => (
        <RailButton key={item.to} item={item} active={isActive(item)} />
      ))}
      <div className="no-drag">
        <ThemeToggle />
      </div>
    </nav>
  );
}
