import { createFileRoute } from "@tanstack/react-router";

import { LogsView } from "@/features/logs/logs-view";

export const Route = createFileRoute("/_shell/logs")({
  component: LogsView,
});
