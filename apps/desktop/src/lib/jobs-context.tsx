import { createContext, use, useEffect, useMemo, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { type Job, streamJobs } from "@/lib/ipc";
import { keys } from "@/lib/queries";

type JobsValue = {
  jobs: Job[];
  active: Job[];
  bySource: Map<string, Job>;
  byModel: Map<string, Job>;
  connected: boolean;
};

const Context = createContext<JobsValue>({
  jobs: [],
  active: [],
  bySource: new Map(),
  byModel: new Map(),
  connected: false,
});

const ACTIVE = new Set(["queued", "running"]);

/**
 * One SSE subscription for the whole app.
 *
 * Every screen that shows progress reads from here rather than opening its own
 * stream, and a finished job invalidates the caches it touched so tables
 * refresh without polling.
 */
export function JobsProvider({ children }: { children: React.ReactNode }) {
  const queryClient = useQueryClient();
  const [jobs, setJobs] = useState<Map<string, Job>>(() => new Map());
  const [connected, setConnected] = useState(false);
  const settled = useRef(new Set<string>());

  useEffect(() => {
    const handle = streamJobs((job) => {
      setConnected(true);
      setJobs((previous) => new Map(previous).set(job.id, job));

      if (ACTIVE.has(job.state) || settled.current.has(job.id)) return;
      settled.current.add(job.id);

      void queryClient.invalidateQueries({ queryKey: keys.jobs });
      if (job.kind === "download") {
        void queryClient.invalidateQueries({ queryKey: keys.models });
      } else {
        void queryClient.invalidateQueries({ queryKey: ["documents"] });
        void queryClient.invalidateQueries({ queryKey: keys.sources });
        void queryClient.invalidateQueries({ queryKey: keys.health });
      }
    });

    return () => {
      void handle.cancel();
    };
  }, [queryClient]);

  const value = useMemo<JobsValue>(() => {
    const list = [...jobs.values()].sort((a, b) =>
      (b.started_at ?? "").localeCompare(a.started_at ?? ""),
    );
    const active = list.filter((job) => ACTIVE.has(job.state));
    const bySource = new Map<string, Job>();
    const byModel = new Map<string, Job>();
    for (const job of active) {
      if (job.source_id) bySource.set(job.source_id, job);
      if (job.model_id) byModel.set(job.model_id, job);
    }
    return { jobs: list, active, bySource, byModel, connected };
  }, [jobs, connected]);

  return <Context value={value}>{children}</Context>;
}

export function useJobs() {
  return use(Context);
}
