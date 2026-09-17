import { createFileRoute } from "@tanstack/react-router";

import { SettingsPanel } from "@/features/settings/settings-panel";

export const Route = createFileRoute("/_shell/settings/$section")({
  component: SettingsPanel,
});
