/**
 * The only way the UI talks to anything outside itself.
 *
 * Features call `api.*` and `streamQuery`; nobody calls `invoke` directly. The
 * Rust shell holds the sidecar's port and session token, so neither ever
 * reaches this side of the boundary.
 */

import { Channel, invoke, isTauri } from "@tauri-apps/api/core";

import type { components } from "./api-types";

export type Schemas = components["schemas"];

export type Health = Schemas["Health"];
export type ProcessStatus = Schemas["ProcessStatus"];
export type IndexStats = Schemas["IndexStats"];
export type Source = Schemas["Source"];
export type SourceCreate = Schemas["SourceCreate"];
export type Document = Schemas["Document"];
export type DocumentList = Schemas["DocumentList"];
export type DocumentContent = Schemas["DocumentContent"];
export type DocumentChunkRef = Schemas["DocumentChunkRef"];
export type DocumentPage = Schemas["DocumentPage"];
export type Job = Schemas["Job"];
export type QueryRequest = Schemas["QueryRequest"];
export type QueryFilters = Schemas["QueryFilters"];
export type RetrievedChunk = Schemas["RetrievedChunk"];
export type Citation = Schemas["Citation"];
export type Connection = Schemas["Connection"];
export type ConnectionInput = Schemas["ConnectionInput"];
export type ConnectionTestRequest = Schemas["ConnectionTestRequest"];
export type ConnectionTestResult = Schemas["ConnectionTestResult"];
export type ModelInventory = Schemas["ModelInventory"];
export type InstalledModel = Schemas["InstalledModel"];
export type HubModel = Schemas["HubModel"];
export type HubSearchResult = Schemas["HubSearchResult"];
export type HubModelDetail = Schemas["HubModelDetail"];
export type HubFile = Schemas["HubFile"];
export type DownloadRequest = Schemas["DownloadRequest"];
export type AppSettings = Schemas["AppSettings"];
export type AppSettingsPatch = Schemas["AppSettingsPatch"];
export type RetrievalSettings = Schemas["RetrievalSettings"];
export type PerformanceSettings = Schemas["PerformanceSettings"];
export type ChatSession = Schemas["ChatSession"];
export type ChatMessage = Schemas["ChatMessage"];
export type EvalSet = Schemas["EvalSet"];
export type StreamEnvelope = Schemas["StreamEnvelope"];
export type StageLatency = NonNullable<NonNullable<StreamEnvelope["done"]>["latency"]>;
export type EvalResult = NonNullable<StreamEnvelope["eval_result"]>;
export type EvalProgress = NonNullable<StreamEnvelope["eval_progress"]>;
export type EvalQuestionResult = EvalResult["questions"][number];
export type EvalMetrics = EvalResult["metrics"];
export type ChatSessionPatch = Schemas["ChatSessionPatch"];
export type ChatProject = Schemas["ChatProject"];
export type ChatProjectPatch = Schemas["ChatProjectPatch"];

/**
 * Literal unions live inline in the OpenAPI schema rather than as named
 * components, so they are derived from the models that carry them. Derived,
 * not retyped: a change on the Python side still lands here.
 */
export type DocumentStatus = Document["status"];
export type Grounding = NonNullable<ChatMessage["grounding"]>;
export type ModelRole = InstalledModel["role"];
export type FitVerdict = NonNullable<HubFile["fit"]>;
export type ConnectionKind = Connection["kind"];
export type QueryMode = NonNullable<QueryRequest["mode"]>;
export type JobState = Job["state"];
export type JobKind = Job["kind"];

/** Measured by the Rust shell, not by the sidecar. */
export type HardwareInfo = {
  os: string;
  arch: string;
  cpu_count: number;
  ram_mb: number;
  vram_mb: number;
  gpu_backend: "metal" | "cuda" | "vulkan" | "cpu";
  gpu_name: string | null;
  profile: "cpu" | "balanced" | "gpu";
};

export type SidecarStatus = {
  name: string;
  role: string;
  state: "stopped" | "starting" | "ready" | "restarting" | "error";
  pid: number | null;
  port: number | null;
  uptime_s: number;
  restarts: number;
  detail: string | null;
};

export type StreamFrame =
  | { kind: "event"; event: string; data: unknown }
  | { kind: "closed"; reason: string }
  | { kind: "failed"; message: string };

export class IpcError extends Error {}

const OUTSIDE_TAURI =
  "This build is running in a plain browser. Start it with `bun tauri dev` so the " +
  "Rust shell can launch the ragcore sidecar.";

async function call<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  if (!isTauri()) throw new IpcError(OUTSIDE_TAURI);
  try {
    return await invoke<T>(command, args);
  } catch (error) {
    throw new IpcError(typeof error === "string" ? error : String(error));
  }
}

function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  return call<T>("api_request", { req: { method, path, body: body ?? null } });
}

function query(params: Record<string, unknown>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") continue;
    search.set(key, String(value));
  }
  const text = search.toString();
  return text ? `?${text}` : "";
}

export const api = {
  health: () => request<Health>("GET", "/health"),

  listSources: () => request<Source[]>("GET", "/sources"),
  addSource: (payload: SourceCreate) => request<Source>("POST", "/sources", payload),
  removeSource: (id: string) => request<void>("DELETE", `/sources/${id}`),
  rescanSource: (id: string) => request<Job>("POST", `/sources/${id}/rescan`),

  listDocuments: (params: {
    source_id?: string;
    status?: DocumentStatus;
    q?: string;
    ext?: string;
    offset?: number;
    limit?: number;
  } = {}) => request<DocumentList>("GET", `/documents${query(params)}`),
  getDocument: (id: string) => request<Document>("GET", `/documents/${id}`),
  getContent: (id: string) => request<DocumentContent>("GET", `/documents/${id}/content`),
  removeDocument: (id: string) => request<void>("DELETE", `/documents/${id}`),
  rebuild: (payload: { source_id?: string; doc_ids?: string[]; full?: boolean }) =>
    request<Job>("POST", "/index/rebuild", payload),

  listJobs: () => request<Job[]>("GET", "/jobs"),
  cancelJob: (id: string) => request<void>("POST", `/jobs/${id}/cancel`),

  listSessions: () => request<ChatSession[]>("GET", "/chats"),
  listProjects: () => request<ChatProject[]>("GET", "/chats/projects"),
  createProject: (payload: { name: string }) =>
    request<ChatProject>("POST", "/chats/projects", payload),
  updateProject: (id: string, payload: ChatProjectPatch) =>
    request<ChatProject>("PATCH", `/chats/projects/${id}`, payload),
  deleteProject: (id: string) => request<void>("DELETE", `/chats/projects/${id}`),
  createSession: (payload: { title?: string; scope_doc_id?: string; project_id?: string } = {}) =>
    request<ChatSession>("POST", "/chats", payload),
  getSession: (id: string) => request<ChatSession>("GET", `/chats/${id}`),
  getMessages: (id: string) => request<ChatMessage[]>("GET", `/chats/${id}/messages`),
  updateSession: (id: string, payload: ChatSessionPatch) =>
    request<ChatSession>("PATCH", `/chats/${id}`, payload),
  renameSession: (id: string, title: string) =>
    request<ChatSession>("PATCH", `/chats/${id}`, { title }),
  deleteSession: (id: string) => request<void>("DELETE", `/chats/${id}`),

  listConnections: () => request<Connection[]>("GET", "/connections"),
  createConnection: (payload: ConnectionInput) =>
    request<Connection>("POST", "/connections", payload),
  updateConnection: (id: string, payload: ConnectionInput) =>
    request<Connection>("PATCH", `/connections/${id}`, payload),
  activateConnection: (id: string) => request<Connection>("POST", `/connections/${id}/activate`),
  deleteConnection: (id: string) => request<void>("DELETE", `/connections/${id}`),
  testConnection: (payload: ConnectionTestRequest) =>
    request<ConnectionTestResult>("POST", "/connections/test", payload),

  models: () => request<ModelInventory>("GET", "/models"),
  hubSearch: (params: { role?: ModelRole; q?: string; sort?: string; limit?: number }) =>
    request<HubSearchResult>("GET", `/models/hub/search${query(params)}`),
  hubRecommended: (role: ModelRole) =>
    request<HubSearchResult>("GET", `/models/hub/recommended${query({ role })}`),
  hubDetail: (repoId: string, role?: ModelRole) =>
    request<HubModelDetail>("GET", `/models/hub/${repoId}${query({ role })}`),
  download: (payload: DownloadRequest) => request<Job>("POST", "/models/download", payload),
  activateModel: (id: string, acceptReindex = false) =>
    request<InstalledModel>(
      "POST",
      `/models/${id}/activate${query({ accept_reindex: acceptReindex })}`,
    ),
  removeModel: (id: string) => request<void>("DELETE", `/models/${id}`),

  settings: () => request<AppSettings>("GET", "/settings"),
  patchSettings: (payload: AppSettingsPatch) =>
    request<AppSettings>("PATCH", "/settings", payload),
  wipe: (keepConnections: boolean) =>
    request<void>("POST", "/settings/wipe", {
      confirm: "DELETE",
      keep_connections: keepConnections,
    }),

  evalSets: () => request<EvalSet[]>("GET", "/eval/sets"),
};

/** Shell-side facts, none of which come from the sidecar. */
export const shell = {
  hardware: () => call<HardwareInfo>("hardware_info"),
  sidecars: () => call<SidecarStatus[]>("sidecar_status"),
  logs: () => call<string[]>("sidecar_logs"),
  restart: () => call<void>("sidecar_restart"),
  keychainSet: (connectionId: string, secret: string) =>
    call<void>("keychain_set", { connectionId, secret }),
  keychainHas: (connectionId: string) => call<boolean>("keychain_has", { connectionId }),
  keychainDelete: (connectionId: string) => call<void>("keychain_delete", { connectionId }),
};

export type StreamHandle = {
  streamId: string;
  done: Promise<void>;
  cancel: () => Promise<void>;
};

/**
 * Open an SSE stream through the shell. The id is generated here so `cancel()`
 * works even before the first frame arrives.
 */
export function openStream(
  options: {
    method?: string;
    path: string;
    body?: unknown;
    cancelPath?: (streamId: string) => string;
  },
  onFrame: (frame: StreamFrame) => void,
): StreamHandle {
  const streamId = `s_${crypto.randomUUID().slice(0, 12)}`;
  const channel = new Channel<StreamFrame>();
  channel.onmessage = onFrame;

  const done = call<void>("api_stream", {
    streamId,
    method: options.method ?? "POST",
    path: options.path,
    body: options.body ?? null,
    channel,
  }).catch((error: unknown) => {
    onFrame({ kind: "failed", message: String(error) });
  });

  return {
    streamId,
    done,
    cancel: () =>
      call<void>("api_cancel", {
        streamId,
        cancelPath: options.cancelPath?.(streamId) ?? null,
      }),
  };
}

/** Frames the /query route emits, in the order it emits them. */
export type QueryEvent =
  | { event: "start"; data: { query_id: string; session_id: string } }
  | { event: "mode"; data: { mode: "local" | "global"; reason: string; scope: string | null } }
  | { event: "sources"; data: { chunks: RetrievedChunk[]; candidates: number; kept: number } }
  | { event: "token"; data: { text: string } }
  | {
      event: "citations";
      data: { citations: Citation[]; dropped: number; grounding: Grounding };
    }
  | {
      event: "done";
      data: {
        message_id: string;
        latency: StageLatency;
        connection_id: string | null;
        remote: boolean;
      };
    }
  | { event: "error"; data: { message: string; retryable: boolean } };

export function streamQuery(
  payload: QueryRequest,
  handlers: {
    onEvent: (event: QueryEvent) => void;
    onClosed?: (reason: string) => void;
    onFailed?: (message: string) => void;
  },
): StreamHandle {
  const queryId = `q_${crypto.randomUUID().slice(0, 12)}`;

  return openStream(
    {
      path: "/query",
      body: { ...payload, query_id: queryId },
      cancelPath: () => `/query/${queryId}/cancel`,
    },
    (frame) => {
      if (frame.kind === "event") {
        handlers.onEvent({ event: frame.event, data: frame.data } as QueryEvent);
      } else if (frame.kind === "closed") {
        handlers.onClosed?.(frame.reason);
      } else {
        handlers.onFailed?.(frame.message);
      }
    },
  );
}

/** Job progress for the Library and Model Manager. One frame per job update. */
export function streamJobs(onJob: (job: Job) => void): StreamHandle {
  return openStream({ method: "GET", path: "/jobs/stream" }, (frame) => {
    if (frame.kind === "event" && frame.event === "job") {
      onJob(frame.data as Job);
    }
  });
}

/** Frames the /eval/run route emits. Dev builds only; the route 403s otherwise. */
export type EvalEvent =
  | { event: "progress"; data: EvalProgress }
  | { event: "result"; data: EvalResult };

export function streamEval(
  payload: { set_name: string; connection_id?: string | null; compare_baseline?: boolean },
  handlers: {
    onEvent: (event: EvalEvent) => void;
    onClosed?: (reason: string) => void;
    onFailed?: (message: string) => void;
  },
): StreamHandle {
  return openStream({ path: "/eval/run", body: payload }, (frame) => {
    if (frame.kind === "event") {
      handlers.onEvent({ event: frame.event, data: frame.data } as EvalEvent);
    } else if (frame.kind === "closed") {
      handlers.onClosed?.(frame.reason);
    } else {
      handlers.onFailed?.(frame.message);
    }
  });
}
