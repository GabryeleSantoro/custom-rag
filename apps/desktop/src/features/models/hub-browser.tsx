import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { DownloadIcon, HeartIcon, SearchIcon, SparklesIcon } from "lucide-react";

import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { count } from "@/lib/format";
import { api, type HubModel, type ModelRole } from "@/lib/ipc";
import { keys } from "@/lib/queries";
import { cn } from "@/lib/utils";

const SORTS = [
  { value: "downloads", label: "Most downloaded" },
  { value: "likes", label: "Most liked" },
  { value: "trendingScore", label: "Trending" },
  { value: "lastModified", label: "Recently updated" },
];

function useDebounced<T>(value: T, delay = 350): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);
  return debounced;
}

function ModelCard({
  model,
  installed,
  onOpen,
}: {
  model: HubModel;
  installed: boolean;
  onOpen: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onOpen}
      className={cn(
        "flex w-full flex-col rounded-lg border bg-card p-3 text-left transition-colors",
        model.recommended
          ? "border-primary/35 hover:border-primary/60"
          : "border-border hover:border-muted-foreground/40",
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate text-[0.8125rem] font-medium">{model.name}</p>
          <p className="truncate font-mono text-[0.625rem] text-muted-foreground">
            {model.author ?? "—"}
          </p>
        </div>
        {installed ? (
          <span className="shrink-0 rounded-full bg-status-ok/12 px-1.5 py-0.5 text-[0.625rem] font-medium text-status-ok">
            Installed
          </span>
        ) : null}
      </div>

      {model.recommended_reason ? (
        <p className="mt-2 line-clamp-2 text-[0.6875rem] leading-[1.45] text-muted-foreground">
          {model.recommended_reason}
        </p>
      ) : null}

      <div className="mt-2.5 flex items-center gap-3 font-mono text-[0.625rem] text-muted-foreground tabular-nums">
        <span className="inline-flex items-center gap-1">
          <DownloadIcon className="size-2.5" />
          {count(model.downloads ?? 0)}
        </span>
        <span className="inline-flex items-center gap-1">
          <HeartIcon className="size-2.5" />
          {count(model.likes ?? 0)}
        </span>
        {model.license ? <span className="truncate">{model.license}</span> : null}
      </div>
    </button>
  );
}

export function HubBrowser({
  role,
  installedRepos,
  onOpen,
}: {
  role: ModelRole;
  installedRepos: Set<string>;
  onOpen: (repoId: string) => void;
}) {
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState("downloads");
  const debounced = useDebounced(search);

  const recommended = useQuery({
    queryKey: keys.hubRecommended(role),
    queryFn: () => api.hubRecommended(role),
    staleTime: 5 * 60_000,
  });

  const params = { role, q: debounced || undefined, sort, limit: 24 };
  const results = useQuery({
    queryKey: keys.hubSearch(params),
    queryFn: () => api.hubSearch(params),
    staleTime: 60_000,
  });

  return (
    <div className="space-y-6 p-5">
      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <SearchIcon className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder={`Search Hugging Face for ${role} models in GGUF`}
            className="h-9 pl-8"
          />
        </div>
        <Select value={sort} onValueChange={setSort}>
          <SelectTrigger className="h-9 w-44 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {SORTS.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {!debounced ? (
        <section>
          <h3 className="flex items-center gap-1.5 text-[0.8125rem] font-semibold tracking-tight">
            <SparklesIcon className="size-3.5 text-primary" />
            Recommended for your hardware
          </h3>
          <p className="mt-0.5 text-[0.6875rem] text-muted-foreground">
            Known-good models whose smallest quantisation this machine can actually run.
          </p>

          <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
            {recommended.isLoading
              ? [0, 1, 2].map((key) => <Skeleton key={key} className="h-24 w-full" />)
              : null}
            {(recommended.data?.items ?? []).map((model) => (
              <ModelCard
                key={model.id}
                model={model}
                installed={installedRepos.has(model.id)}
                onOpen={() => onOpen(model.id)}
              />
            ))}
          </div>
        </section>
      ) : null}

      <section>
        <h3 className="text-[0.8125rem] font-semibold tracking-tight">
          {debounced ? `Results for “${debounced}”` : "Popular on Hugging Face"}
        </h3>

        {results.isError ? (
          <p className="mt-2 text-xs text-status-error">
            {(results.error as Error).message}
          </p>
        ) : null}

        <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
          {results.isLoading
            ? [0, 1, 2, 3, 4, 5].map((key) => <Skeleton key={key} className="h-24 w-full" />)
            : null}
          {(results.data?.items ?? []).map((model) => (
            <ModelCard
              key={model.id}
              model={model}
              installed={installedRepos.has(model.id)}
              onOpen={() => onOpen(model.id)}
            />
          ))}
        </div>

        {results.data?.items.length === 0 ? (
          <p className="mt-3 text-xs text-muted-foreground">
            Nothing matched. GGUF is required, so repositories that only publish safetensors are
            filtered out.
          </p>
        ) : null}
      </section>
    </div>
  );
}
