import { Link, useParams } from "@tanstack/react-router";

import { ContextSidebar } from "@/components/shell/context-sidebar";
import { cn } from "@/lib/utils";

export const SETTINGS_SECTIONS = [
  {
    slug: "connections",
    label: "Connections",
    blurb: "Where answers are written. Keys live in the OS keychain.",
  },
  {
    slug: "retrieval",
    label: "Retrieval",
    blurb: "How passages are found, ranked and packed into the prompt.",
  },
  {
    slug: "performance",
    label: "Performance",
    blurb: "Batch sizes and GPU offload, defaulted from the detected hardware.",
  },
  { slug: "storage", label: "Storage", blurb: "Where data lives, and how to remove it." },
] as const;

export function SettingsSidebar() {
  const params = useParams({ strict: false }) as { section?: string };

  return (
    <ContextSidebar title="Settings">
      <nav className="flex flex-col gap-0.5 pt-1">
        {SETTINGS_SECTIONS.map((section) => (
          <Link
            key={section.slug}
            to="/settings/$section"
            params={{ section: section.slug }}
            className={cn(
              "rounded-md px-2 py-1.5 text-[0.8125rem] text-sidebar-foreground transition-colors",
              "hover:bg-sidebar-accent",
              params.section === section.slug && "bg-sidebar-accent font-medium",
            )}
          >
            {section.label}
          </Link>
        ))}
      </nav>
    </ContextSidebar>
  );
}
