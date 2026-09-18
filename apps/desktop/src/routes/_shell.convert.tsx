import { createFileRoute } from "@tanstack/react-router";

import { ConverterView } from "@/features/converter/converter-view";

export const Route = createFileRoute("/_shell/convert")({
  component: ConverterView,
});
