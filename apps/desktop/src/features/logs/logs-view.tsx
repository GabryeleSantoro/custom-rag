import { useState } from "react";
import { useTranslation } from "react-i18next";
import { useQuery } from "@tanstack/react-query";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import { ClipboardCheckIcon, ClipboardCopyIcon, FolderOpenIcon } from "lucide-react";
import { toast } from "sonner";

import { LogPane, levelOf } from "@/components/log-pane";
import { Page, PageBody, PageHeader } from "@/components/shell/page";
import { Button } from "@/components/ui/button";
import { errorText } from "@/lib/errors";
import { shell } from "@/lib/ipc";
import { logsQuery } from "@/lib/queries";

/**
 * Everything the app logged since it started: shell events, proxy failures and
 * stream durations, and every line the core prints. The same lines go to a file
 * that survives restarts, one click away.
 */
export function LogsView() {
  const { t } = useTranslation();
  const logs = useQuery(logsQuery);
  const logFile = useQuery({ queryKey: ["log-file"], queryFn: shell.logFile, staleTime: Infinity });
  const [copied, setCopied] = useState(false);
  const lines = logs.data ?? [];
  const errors = lines.filter((line) => levelOf(line) === "error").length;

  async function copyAll() {
    try {
      await navigator.clipboard.writeText(lines.join("\n"));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (error) {
      toast.error(t("logs.copyFailed"), { description: errorText(error) });
    }
  }

  async function openFolder() {
    if (!logFile.data) return;
    try {
      await revealItemInDir(logFile.data);
    } catch (error) {
      toast.error(t("logs.openFailed"), { description: errorText(error) });
    }
  }

  return (
    <Page>
      <PageHeader
        title={t("nav.logs")}
        description={
          t("logs.lineCount", { count: lines.length }) +
          (errors ? ` · ${t("logs.errorCount", { count: errors })}` : "")
        }
        actions={
          <>
            <Button variant="secondary" size="sm" className="h-8" onClick={copyAll} disabled={!lines.length}>
              {copied ? <ClipboardCheckIcon className="size-3.5" /> : <ClipboardCopyIcon className="size-3.5" />}
              {copied ? t("logs.copied") : t("logs.copyAll")}
            </Button>
            <Button
              variant="secondary"
              size="sm"
              className="h-8"
              onClick={openFolder}
              disabled={!logFile.data}
              title={logFile.data ?? t("logs.noFile")}
            >
              <FolderOpenIcon className="size-3.5" />
              {t("logs.openFile")}
            </Button>
          </>
        }
      />
      <PageBody>
        <div className="flex h-full flex-col p-5">
          <LogPane lines={lines} fill />
          <p className="mt-2 font-mono text-[0.625rem] text-muted-foreground">
            {t("logs.footer", { path: logFile.data ?? t("logs.noFile") })}
          </p>
        </div>
      </PageBody>
    </Page>
  );
}
