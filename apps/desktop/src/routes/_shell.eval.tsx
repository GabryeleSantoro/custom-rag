import { createFileRoute } from "@tanstack/react-router";

import { EvalView } from "@/features/eval/eval-view";

export const Route = createFileRoute("/_shell/eval")({
  component: EvalView,
});
