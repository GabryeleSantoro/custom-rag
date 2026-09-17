import { createFileRoute } from "@tanstack/react-router";

import { DiagnosticsView } from "@/features/diagnostics/diagnostics-view";

export const Route = createFileRoute("/_shell/diagnostics")({
  component: DiagnosticsView,
});
