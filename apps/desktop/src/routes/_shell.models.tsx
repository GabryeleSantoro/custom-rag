import { createFileRoute } from "@tanstack/react-router";

import { ModelsView } from "@/features/models/models-view";

export const Route = createFileRoute("/_shell/models")({
  component: ModelsView,
});
