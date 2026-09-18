import { useRef, useState } from "react";
import { Link } from "@tanstack/react-router";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { open } from "@tauri-apps/plugin-dialog";
import {
  BookOpenTextIcon,
  CheckCircle2Icon,
  ChevronRightIcon,
  FileUpIcon,
  Globe2Icon,
  LoaderCircleIcon,
  LockKeyholeIcon,
  SearchIcon,
  SparklesIcon,
  XIcon,
  WandSparklesIcon,
} from "lucide-react";

import { Page, PageBody, PageHeader } from "@/components/shell/page";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import {
  api,
  streamSlideConversion,
  type ConversionEvent,
  type Document,
  type StreamHandle,
  type WebResearchResult,
} from "@/lib/ipc";
import { connectionsQuery, keys } from "@/lib/queries";
import { cn } from "@/lib/utils";

type ConversionStatus = "idle" | "researching" | "generating" | "saved" | "error";
type LocalSlide = { path: string; title: string; ext: string };

const STATUS_LABEL: Record<ConversionStatus, string> = {
  idle: "Pronto",
  researching: "Ricerca web in corso",
  generating: "Scrittura del testo",
  saved: "Salvato nella libreria globale",
  error: "Conversione non riuscita",
};

function readableExtension(document: Document) {
  return document.ext.replace(".", "").toUpperCase() || "FILE";
}

function localSlideFromPath(path: string): LocalSlide {
  const filename = path.split(/[\\/]/).pop() ?? path;
  const dot = filename.lastIndexOf(".");
  return {
    path,
    title: dot > 0 ? filename.slice(0, dot) : filename,
    ext: dot > 0 ? filename.slice(dot + 1).toUpperCase() : "FILE",
  };
}

function ConverterStatus({ status }: { status: ConversionStatus }) {
  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground">
      {status === "researching" || status === "generating" ? (
        <LoaderCircleIcon className="size-3.5 animate-spin text-primary" />
      ) : status === "saved" ? (
        <CheckCircle2Icon className="size-3.5 text-status-ok" />
      ) : status === "error" ? (
        <span className="size-2 rounded-full bg-status-error" />
      ) : (
        <span className="size-2 rounded-full bg-status-idle" />
      )}
      <span>{STATUS_LABEL[status]}</span>
    </div>
  );
}

function SlideRow({
  document,
  checked,
  disabled,
  onCheckedChange,
}: {
  document: Document;
  checked: boolean;
  disabled: boolean;
  onCheckedChange: (checked: boolean) => void;
}) {
  return (
    <label
      className={cn(
        "group flex cursor-pointer items-start gap-3 border-b border-border px-4 py-3 last:border-0",
        "transition-colors hover:bg-muted/45",
        disabled && "cursor-not-allowed opacity-55",
      )}
    >
      <Checkbox
        checked={checked}
        disabled={disabled}
        onCheckedChange={(value) => onCheckedChange(value === true)}
        className="mt-0.5"
      />
      <BookOpenTextIcon className="mt-0.5 size-4 shrink-0 text-primary/75" />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm font-medium">{document.title}</span>
        <span className="mt-0.5 block truncate font-mono text-[0.65rem] text-muted-foreground">
          {readableExtension(document)} · {document.n_pages ?? "—"} pagine · {document.path}
        </span>
      </span>
      <ChevronRightIcon className="mt-0.5 size-3.5 text-muted-foreground/45 transition-transform group-hover:translate-x-0.5" />
    </label>
  );
}

export function ConverterView() {
  const queryClient = useQueryClient();
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [uploadedSlides, setUploadedSlides] = useState<LocalSlide[]>([]);
  const [researchQuery, setResearchQuery] = useState("");
  const [outputTitle, setOutputTitle] = useState("");
  const [language, setLanguage] = useState<"it" | "en">("it");
  const [depth, setDepth] = useState<"standard" | "deep">("deep");
  const [status, setStatus] = useState<ConversionStatus>("idle");
  const [output, setOutput] = useState("");
  const [researchResults, setResearchResults] = useState<WebResearchResult[]>([]);
  const [researchWarning, setResearchWarning] = useState<string | null>(null);
  const [savedPath, setSavedPath] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const stream = useRef<StreamHandle | null>(null);

  const connections = useQuery(connectionsQuery);
  const documents = useQuery({
    queryKey: keys.documents({ limit: 500 }),
    queryFn: () => api.listDocuments({ limit: 500 }),
  });
  const active = connections.data?.find((connection) => connection.active);
  const items = documents.data?.items ?? [];
  const selected = items.filter((document) => selectedIds.includes(document.id));
  const selectedCount = selected.length + uploadedSlides.length;
  const canConvert = Boolean(
    active && selectedCount && status !== "researching" && status !== "generating",
  );

  const toggle = (id: string, checked: boolean) => {
    setSelectedIds((current) =>
      checked ? [...current, id] : current.filter((selectedId) => selectedId !== id),
    );
  };

  const pickSlides = async () => {
    try {
      const picked = await open({
        directory: false,
        multiple: true,
        title: "Carica slide da convertire",
        filters: [{ name: "Slide", extensions: ["pdf", "pptx"] }],
      });
      const paths = Array.isArray(picked) ? picked : picked ? [picked] : [];
      if (!paths.length) return;
      setUploadedSlides((current) => {
        const known = new Set(current.map((file) => file.path));
        return [...current, ...paths.filter((path) => !known.has(path)).map(localSlideFromPath)];
      });
      setError(null);
    } catch (cause) {
      setStatus("error");
      setError(String(cause));
    }
  };

  const handleEvent = (event: ConversionEvent) => {
    if (event.event === "conversion_research") {
      setStatus("researching");
      setResearchResults(event.data.results);
      setResearchWarning(event.data.warning ?? null);
    } else if (event.event === "token") {
      setStatus("generating");
      setOutput((current) => current + event.data.text);
    } else if (event.event === "conversion_saved") {
      setSavedPath(event.data.path);
    } else if (event.event === "conversion_done") {
      setStatus("saved");
      setSavedPath(event.data.path);
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
      void queryClient.invalidateQueries({ queryKey: keys.sources });
    } else if (event.event === "error") {
      setStatus("error");
      setError(event.data.message);
    }
  };

  const convert = () => {
    if (!canConvert) return;
    setStatus("researching");
    setOutput("");
    setSavedPath(null);
    setResearchResults([]);
    setResearchWarning(null);
    setError(null);
    stream.current = streamSlideConversion(
      {
        slide_ids: selectedIds,
        file_paths: uploadedSlides.map((file) => file.path),
        research_query: researchQuery.trim() || null,
        output_title: outputTitle.trim() || null,
        language,
        depth,
      },
      {
        onEvent: handleEvent,
        onFailed: (message) => {
          setStatus("error");
          setError(message);
        },
      },
    );
  };

  const stop = () => {
    void stream.current?.cancel();
    setStatus("idle");
  };

  return (
    <Page>
      <PageHeader
        title="Slide → testo"
        description={active ? `${active.name} · ${active.model_id}` : "Nessun modello collegato"}
        actions={
          <>
            <ConverterStatus status={status} />
            <Button asChild variant="ghost" size="sm" className="h-8">
              <Link to="/chat">Apri chat</Link>
            </Button>
          </>
        }
      />

      <PageBody>
        <main className="mx-auto w-full max-w-6xl space-y-6 px-6 py-7">
          <section className="max-w-3xl">
            <p className="font-mono text-[0.65rem] tracking-[0.18em] text-primary uppercase">
              Research workspace / 01
            </p>
            <h1 className="mt-2 text-3xl font-semibold tracking-[-0.045em] text-balance sm:text-4xl">
              Dalle slide a un testo che regge anche fuori dalla presentazione.
            </h1>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-muted-foreground">
              Seleziona le slide indicizzate, aggiungi una direzione di ricerca e lascia che il modello colleghi i concetti in un documento Markdown completo, con le fonti web in coda.
            </p>
          </section>

          {!active ? (
            <Alert variant="destructive">
              <LockKeyholeIcon />
              <AlertTitle>Collega un modello per abilitare la conversione</AlertTitle>
              <AlertDescription>
                La funzione usa il modello di generazione attivo e rimane disabilitata finché non esiste una connessione.
                <Link to="/settings/$section" params={{ section: "models" }} className="mt-2 inline-flex font-medium underline underline-offset-4">
                  Vai alle connessioni
                </Link>
              </AlertDescription>
            </Alert>
          ) : null}

          <div className="grid gap-5 xl:grid-cols-[minmax(0,1.25fr)_minmax(20rem,0.75fr)]">
            <section className="overflow-hidden rounded-xl border border-border bg-card">
              <div className="flex flex-wrap items-start justify-between gap-4 border-b border-border px-4 py-4">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="grid size-6 place-items-center rounded-md bg-primary/12 font-mono text-[0.65rem] font-semibold text-primary">A</span>
                    <h2 className="text-sm font-semibold">Sorgenti slide</h2>
                  </div>
                  <p className="mt-1 pl-8 text-xs text-muted-foreground">Scegli dalla libreria oppure carica slide solo per questa conversione.</p>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="secondary" className="shrink-0 font-mono text-[0.65rem]">
                    {selectedCount} selezionati
                  </Badge>
                  <Button
                    type="button"
                    variant="secondary"
                    size="sm"
                    className="h-7"
                    disabled={!active || status === "researching" || status === "generating"}
                    onClick={pickSlides}
                  >
                    <FileUpIcon className="size-3.5" />
                    Carica slide
                  </Button>
                </div>
              </div>

              {documents.isLoading ? (
                <div className="space-y-2 p-4">
                  <Skeleton className="h-12 w-full" />
                  <Skeleton className="h-12 w-full" />
                  <Skeleton className="h-12 w-full" />
                </div>
              ) : items.length === 0 && uploadedSlides.length === 0 ? (
                <div className="grid min-h-48 place-items-center p-8 text-center">
                  <div>
                    <BookOpenTextIcon className="mx-auto size-7 text-muted-foreground/50" />
                    <p className="mt-3 text-sm font-medium">Nessun documento indicizzato</p>
                    <p className="mt-1 max-w-xs text-xs leading-5 text-muted-foreground">Aggiungi una cartella dalla Libreria per rendere disponibili le slide.</p>
                  </div>
                </div>
              ) : (
                <div className="max-h-80 overflow-auto">
                  {uploadedSlides.map((file) => (
                    <div key={file.path} className="flex items-start gap-3 border-b border-primary/15 bg-primary/5 px-4 py-3">
                      <FileUpIcon className="mt-0.5 size-4 shrink-0 text-primary" />
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-sm font-medium">{file.title}</span>
                        <span className="mt-0.5 block truncate font-mono text-[0.65rem] text-muted-foreground">
                          {file.ext} · solo conversione · {file.path}
                        </span>
                      </span>
                      <button
                        type="button"
                        className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                        aria-label={`Rimuovi ${file.title}`}
                        onClick={() => setUploadedSlides((current) => current.filter((item) => item.path !== file.path))}
                      >
                        <XIcon className="size-3.5" />
                      </button>
                    </div>
                  ))}
                  {items.map((document) => (
                    <SlideRow
                      key={document.id}
                      document={document}
                      checked={selectedIds.includes(document.id)}
                      disabled={!active || status === "researching" || status === "generating"}
                      onCheckedChange={(checked) => toggle(document.id, checked)}
                    />
                  ))}
                </div>
              )}
            </section>

            <section className="rounded-xl border border-border bg-card p-5">
              <div className="flex items-center gap-2">
                <span className="grid size-6 place-items-center rounded-md bg-primary/12 font-mono text-[0.65rem] font-semibold text-primary">B</span>
                <h2 className="text-sm font-semibold">Brief di ricerca</h2>
              </div>
              <div className="mt-5 space-y-4">
                <div className="space-y-1.5">
                  <Label htmlFor="research-query">Cosa vuoi approfondire?</Label>
                  <Textarea
                    id="research-query"
                    value={researchQuery}
                    onChange={(event) => setResearchQuery(event.target.value)}
                    placeholder="Es. stato dell'arte, casi d'uso e implicazioni per il mercato…"
                    className="min-h-24 resize-none text-sm"
                    disabled={!active || status === "researching" || status === "generating"}
                  />
                  <p className="text-[0.68rem] leading-4 text-muted-foreground">Lascia vuoto per ricavare la ricerca dal titolo delle slide.</p>
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="output-title">Titolo del documento</Label>
                  <Input
                    id="output-title"
                    value={outputTitle}
                    onChange={(event) => setOutputTitle(event.target.value)}
                    placeholder={selected[0]?.title ?? uploadedSlides[0]?.title ?? "Titolo del testo"}
                    disabled={!active || status === "researching" || status === "generating"}
                  />
                </div>
                <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2">
                  <div className="space-y-1.5">
                    <Label>Lingua</Label>
                    <Select value={language} onValueChange={(value) => setLanguage(value as "it" | "en")}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent><SelectItem value="it">Italiano</SelectItem><SelectItem value="en">English</SelectItem></SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-1.5">
                    <Label>Profondità</Label>
                    <Select value={depth} onValueChange={(value) => setDepth(value as "standard" | "deep")}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent><SelectItem value="deep">Approfondita</SelectItem><SelectItem value="standard">Standard</SelectItem></SelectContent>
                    </Select>
                  </div>
                </div>
                <Button className="mt-1 w-full" disabled={!canConvert} onClick={convert}>
                  <WandSparklesIcon className="size-4" />
                  Genera testo Markdown
                </Button>
                {status === "researching" || status === "generating" ? (
                  <Button variant="ghost" size="sm" className="w-full" onClick={stop}>Interrompi</Button>
                ) : null}
              </div>
            </section>
          </div>

          {error ? (
            <Alert variant="destructive"><SearchIcon /><AlertTitle>La conversione si è fermata</AlertTitle><AlertDescription>{error}</AlertDescription></Alert>
          ) : null}

          {(output || researchResults.length || savedPath) ? (
            <section className="grid gap-5 xl:grid-cols-[minmax(0,1.5fr)_minmax(18rem,0.5fr)]">
              <article className="overflow-hidden rounded-xl border border-border bg-card">
                <div className="flex items-center justify-between border-b border-border px-4 py-3">
                  <div className="flex items-center gap-2"><SparklesIcon className="size-4 text-primary" /><h2 className="text-sm font-semibold">Anteprima del testo</h2></div>
                  <Badge variant="outline" className="font-mono text-[0.6rem]">.md</Badge>
                </div>
                <div className="max-h-[28rem] overflow-auto p-5">
                  {output ? <pre className="selectable whitespace-pre-wrap font-sans text-sm leading-7 text-foreground">{output}</pre> : <div className="space-y-3"><Skeleton className="h-5 w-3/4" /><Skeleton className="h-4 w-full" /><Skeleton className="h-4 w-11/12" /><Skeleton className="h-4 w-2/3" /></div>}
                </div>
              </article>

              <aside className="space-y-5">
                <div className="rounded-xl border border-border bg-card p-4">
                  <div className="flex items-center gap-2"><Globe2Icon className="size-4 text-primary" /><h2 className="text-sm font-semibold">Ricerca web</h2></div>
                  <p className="mt-1 text-xs leading-5 text-muted-foreground">Le fonti consultate restano collegate al file Markdown generato.</p>
                  {researchWarning ? <p className="mt-3 rounded-md bg-status-warn/10 px-2.5 py-2 text-xs leading-5 text-status-warn">{researchWarning}</p> : null}
                  <div className="mt-3 space-y-2">
                    {researchResults.map((result) => <a key={result.url} href={result.url} target="_blank" rel="noreferrer" className="block rounded-md border border-border/70 p-2.5 transition-colors hover:border-primary/45 hover:bg-muted/30"><p className="line-clamp-2 text-xs font-medium">{result.title}</p><p className="mt-1 truncate font-mono text-[0.6rem] text-muted-foreground">{result.url}</p></a>)}
                    {!researchResults.length && !researchWarning ? <p className="text-xs text-muted-foreground">In attesa della ricerca.</p> : null}
                  </div>
                </div>
                {savedPath ? <div className="rounded-xl border border-status-ok/25 bg-status-ok/6 p-4"><div className="flex items-center gap-2 text-status-ok"><CheckCircle2Icon className="size-4" /><p className="text-sm font-semibold">File pronto</p></div><p className="mt-2 break-all font-mono text-[0.65rem] leading-5 text-muted-foreground">{savedPath}</p><Link to="/library" className="mt-3 inline-flex text-xs font-medium text-primary underline underline-offset-4">Apri nella Libreria</Link></div> : null}
              </aside>
            </section>
          ) : null}

          <div className="flex items-center gap-2 border-t border-border pt-4 text-[0.68rem] text-muted-foreground">
            <Globe2Icon className="size-3.5" /> La ricerca web viene usata come contesto aggiuntivo; il risultato viene salvato sempre come file Markdown nella cartella globale.
          </div>
        </main>
      </PageBody>
    </Page>
  );
}
