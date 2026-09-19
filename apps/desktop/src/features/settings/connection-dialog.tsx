import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2Icon, Loader2Icon, XCircleIcon } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ms } from "@/lib/format";
import {
  api,
  shell,
  type Connection,
  type ConnectionKind,
  type ConnectionTestResult,
} from "@/lib/ipc";
import { keys } from "@/lib/queries";

type ProviderId = "openrouter" | "openai" | "anthropic" | "local-inapp" | "custom";

/** A preset prefills kind + base_url; "custom" is the only one with an editable URL. */
const PROVIDERS: { id: ProviderId; label: string; kind: ConnectionKind; baseUrl: string | null; hint: string }[] = [
  {
    id: "openrouter",
    label: "OpenRouter",
    kind: "openai-compatible",
    baseUrl: "https://openrouter.ai/api/v1",
    hint: "Routes to any model OpenRouter offers",
  },
  {
    id: "openai",
    label: "OpenAI",
    kind: "openai-compatible",
    baseUrl: "https://api.openai.com/v1",
    hint: "GPT models, native OpenAI API",
  },
  { id: "anthropic", label: "Anthropic", kind: "anthropic", baseUrl: null, hint: "Claude models, native API" },
  {
    id: "local-inapp",
    label: "In-app model",
    kind: "local-inapp",
    baseUrl: null,
    hint: "A GGUF downloaded here, served by the app itself",
  },
  {
    id: "custom",
    label: "Custom / localhost",
    kind: "openai-compatible",
    baseUrl: null,
    hint: "LM Studio, Ollama, llama.cpp, vLLM, or any other OpenAI-compatible server",
  },
];

function isRemoteUrl(url: string) {
  return !/localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\]/.test(url);
}

function matchProvider(kind: ConnectionKind, base_url: string | null | undefined) {
  const normalized = (base_url ?? "").replace(/\/+$/, "");
  return (
    PROVIDERS.find((p) => p.baseUrl && p.kind === kind && p.baseUrl.replace(/\/+$/, "") === normalized) ??
    PROVIDERS.find((p) => p.baseUrl === null && p.kind === kind) ??
    PROVIDERS[PROVIDERS.length - 1]
  );
}

const BLANK = {
  name: "",
  kind: "openai-compatible" as ConnectionKind,
  base_url: "http://localhost:1234/v1",
  model_id: "",
  max_output_tokens: 1024,
  thinking: "off" as const,
  is_remote: false,
  provider_sort: null as "price" | "throughput" | "latency" | null,
  provider_order: null as string[] | null,
};

const SORT_MODES: { value: "auto" | "price" | "throughput" | "latency"; label: string }[] = [
  { value: "auto", label: "Auto (OpenRouter default)" },
  { value: "price", label: "Cheapest" },
  { value: "throughput", label: "Fastest (throughput)" },
  { value: "latency", label: "Fastest (first token)" },
];

function TestResult({ result }: { result: ConnectionTestResult }) {
  const Icon = result.ok ? CheckCircle2Icon : XCircleIcon;
  return (
    <div
      className={`flex items-start gap-2 rounded-md border px-2.5 py-2 text-[0.6875rem] ${
        result.ok
          ? "border-status-ok/30 bg-status-ok/8 text-status-ok"
          : "border-status-warn/30 bg-status-warn/8 text-status-warn"
      }`}
    >
      <Icon className="mt-px size-3.5 shrink-0" />
      <div className="space-y-0.5">
        <p>
          {result.reachable ? "Reachable" : "Not reachable"}
          {result.latency_ms != null ? ` in ${ms(result.latency_ms)}` : ""}
          {result.model_found ? " · model found" : ""}
          {result.streaming ? " · streaming supported" : ""}
        </p>
        {result.error ? <p className="opacity-80">{result.error}</p> : null}
      </div>
    </div>
  );
}

export function ConnectionDialog({
  connection,
  trigger,
}: {
  connection?: Connection;
  trigger: React.ReactNode;
}) {
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState(() => ({ ...BLANK, ...(connection ?? {}) }));
  const [provider, setProvider] = useState(() => matchProvider(form.kind, form.base_url));
  const [providerOrderText, setProviderOrderText] = useState(() =>
    (form.provider_order ?? []).join(", "),
  );
  const [apiKey, setApiKey] = useState("");
  const [result, setResult] = useState<ConnectionTestResult | null>(null);

  const set = <K extends keyof typeof form>(key: K, value: (typeof form)[K]) =>
    setForm((current) => ({ ...current, [key]: value }));

  const test = useMutation({
    mutationFn: () =>
      api.testConnection({
        kind: form.kind,
        base_url: form.base_url,
        model_id: form.model_id,
        api_key: apiKey || null,
      }),
    onSuccess: setResult,
    onError: (error: Error) => toast.error("Test failed", { description: error.message }),
  });

  const save = useMutation({
    mutationFn: async () => {
      const payload = {
        name: form.name || form.model_id,
        kind: form.kind,
        base_url: form.kind === "local-inapp" || form.kind === "anthropic" ? null : form.base_url,
        model_id: form.model_id,
        max_output_tokens: form.max_output_tokens,
        thinking: form.thinking,
        is_remote: form.is_remote,
        api_key: apiKey || null,
        provider_sort: provider.id === "openrouter" ? form.provider_sort : null,
        provider_order:
          provider.id === "openrouter" && providerOrderText.trim()
            ? providerOrderText
                .split(",")
                .map((entry) => entry.trim())
                .filter(Boolean)
            : null,
      };
      const saved = connection
        ? await api.updateConnection(connection.id, payload)
        : await api.createConnection(payload);
      // The keychain is the durable copy; the sidecar got a memory-only one in
      // the payload above, because the sidecar is what calls the provider.
      if (apiKey) await shell.keychainSet(saved.id, apiKey);
      return saved;
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: keys.connections });
      setOpen(false);
      setApiKey("");
      toast.success(connection ? "Connection updated" : "Connection added");
    },
    onError: (error: Error) => toast.error("Could not save", { description: error.message }),
  });

  const needsUrl = provider.id === "custom";

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>{trigger}</DialogTrigger>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{connection ? "Edit connection" : "Add a connection"}</DialogTitle>
          <DialogDescription>
            The model writes the answer. Embedding and reranking always stay on this device,
            whatever you connect here.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label htmlFor="conn-kind">Provider</Label>
              <Select
                value={provider.id}
                onValueChange={(value) => {
                  const next = PROVIDERS.find((entry) => entry.id === value)!;
                  setProvider(next);
                  setForm((current) => ({
                    ...current,
                    kind: next.kind,
                    base_url: next.baseUrl ?? (next.id === "custom" ? current.base_url : null),
                    is_remote:
                      next.id === "local-inapp"
                        ? false
                        : next.id === "custom"
                          ? isRemoteUrl(current.base_url ?? "")
                          : true,
                  }));
                }}
              >
                <SelectTrigger id="conn-kind">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {PROVIDERS.map((entry) => (
                    <SelectItem key={entry.id} value={entry.id}>
                      {entry.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-[0.6875rem] text-muted-foreground">
                {provider.hint}
                {provider.baseUrl ? ` · ${provider.baseUrl}` : ""}
              </p>
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="conn-name">Name</Label>
              <Input
                id="conn-name"
                value={form.name}
                placeholder="LM Studio"
                onChange={(event) => set("name", event.target.value)}
              />
            </div>
          </div>

          {provider.id === "openrouter" ? (
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-1.5">
                <Label htmlFor="conn-provider-sort">Provider routing</Label>
                <Select
                  value={form.provider_sort ?? "auto"}
                  onValueChange={(value) =>
                    set(
                      "provider_sort",
                      value === "auto" ? null : (value as "price" | "throughput" | "latency"),
                    )
                  }
                >
                  <SelectTrigger id="conn-provider-sort">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {SORT_MODES.map((mode) => (
                      <SelectItem key={mode.value} value={mode.value}>
                        {mode.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="conn-provider-order">Preferred providers</Label>
                <Input
                  id="conn-provider-order"
                  value={providerOrderText}
                  placeholder="e.g. together, fireworks"
                  onChange={(event) => setProviderOrderText(event.target.value)}
                />
                <p className="text-[0.6875rem] text-muted-foreground">
                  Comma-separated, tried in order. Leave empty to let OpenRouter pick.
                </p>
              </div>
            </div>
          ) : null}

          {needsUrl ? (
            <div className="space-y-1.5">
              <Label htmlFor="conn-url">Base URL</Label>
              <Input
                id="conn-url"
                value={form.base_url ?? ""}
                placeholder="http://localhost:1234/v1"
                onChange={(event) => {
                  set("base_url", event.target.value);
                  set(
                    "is_remote",
                    !/localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\]/.test(event.target.value),
                  );
                }}
              />
              <p className="text-[0.6875rem] text-muted-foreground">
                {form.is_remote
                  ? "Remote: retrieved passages will leave this device."
                  : "Local: nothing leaves this device."}
              </p>
            </div>
          ) : null}

          <div className="grid gap-4 sm:grid-cols-3">
            <div className="space-y-1.5 sm:col-span-2">
              <Label htmlFor="conn-model">Model ID</Label>
              <Input
                id="conn-model"
                value={form.model_id}
                placeholder="qwen3-8b-instruct"
                onChange={(event) => set("model_id", event.target.value)}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="conn-thinking">Thinking</Label>
              <Select
                value={form.thinking}
                onValueChange={(value) => set("thinking", value as typeof form.thinking)}
              >
                <SelectTrigger id="conn-thinking">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {["off", "low", "medium", "high"].map((level) => (
                    <SelectItem key={level} value={level}>
                      {level}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label htmlFor="conn-output">Max output tokens</Label>
              <Input
                id="conn-output"
                type="number"
                value={form.max_output_tokens}
                onChange={(event) => set("max_output_tokens", Number(event.target.value) || 0)}
              />
            </div>
          </div>

          {form.kind !== "local-inapp" ? (
            <div className="space-y-1.5">
              <Label htmlFor="conn-key">
                API key {connection?.has_api_key ? "(stored — type to replace)" : ""}
              </Label>
              <Input
                id="conn-key"
                type="password"
                value={apiKey}
                placeholder={connection?.has_api_key ? "••••••••" : "Leave empty for local servers"}
                onChange={(event) => setApiKey(event.target.value)}
              />
              <p className="text-[0.6875rem] text-muted-foreground">
                Stored in the OS keychain. It is never written to the index, settings or logs.
              </p>
            </div>
          ) : null}

          {result ? <TestResult result={result} /> : null}
        </div>

        <DialogFooter className="sm:justify-between">
          <Button variant="secondary" onClick={() => test.mutate()} disabled={test.isPending}>
            {test.isPending ? <Loader2Icon className="size-4 animate-spin" /> : null}
            Test connection
          </Button>
          <div className="flex gap-2">
            <Button variant="ghost" onClick={() => setOpen(false)}>
              Cancel
            </Button>
            <Button
              disabled={!form.model_id.trim() || save.isPending}
              onClick={() => save.mutate()}
            >
              Save
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
