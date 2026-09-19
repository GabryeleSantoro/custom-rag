import { useCallback, useEffect, useState } from "react";
import { getVersion } from "@tauri-apps/api/app";
import { relaunch } from "@tauri-apps/plugin-process";
import { check } from "@tauri-apps/plugin-updater";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";

const AUTO_KEY = "updates.auto";

const autoUpdateEnabled = () => localStorage.getItem(AUTO_KEY) !== "off";

async function runUpdate(silent: boolean) {
  const update = await check();
  if (!update) {
    if (!silent) toast.success("You are on the latest version");
    return;
  }
  toast.info(`Version ${update.version} is downloading`, { description: update.body });
  await update.downloadAndInstall();
  await relaunch();
}

/** Checks once at startup unless the user turned auto-updates off. */
export function useAutoUpdate() {
  useEffect(() => {
    if (!autoUpdateEnabled()) return;
    void runUpdate(true).catch(() => undefined);
  }, []);
}

export function UpdatesSection() {
  const [version, setVersion] = useState("");
  const [auto, setAuto] = useState(autoUpdateEnabled);
  const [checking, setChecking] = useState(false);

  useEffect(() => {
    void getVersion().then(setVersion);
  }, []);

  const checkNow = useCallback(() => {
    setChecking(true);
    runUpdate(false)
      .catch((error: unknown) =>
        toast.error("Update check failed", { description: String(error) }),
      )
      .finally(() => setChecking(false));
  }, []);

  return (
    <div className="space-y-6">
      <div className="grid gap-1.5 sm:grid-cols-[14rem_1fr] sm:items-baseline sm:gap-6">
        <div>
          <Label className="text-[0.8125rem]">Automatic updates</Label>
          <p className="mt-0.5 text-[0.6875rem] text-muted-foreground">
            Checks at launch and installs in the background, then restarts.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Switch
            checked={auto}
            onCheckedChange={(checked) => {
              setAuto(checked);
              localStorage.setItem(AUTO_KEY, checked ? "on" : "off");
            }}
          />
          <span className="text-[0.8125rem]">{auto ? "Enabled" : "Disabled"}</span>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <Button disabled={checking} onClick={checkNow}>
          {checking ? "Checking…" : "Check for updates"}
        </Button>
        <span className="text-[0.6875rem] text-muted-foreground">
          {version ? `Current version ${version}` : null}
        </span>
      </div>
    </div>
  );
}
