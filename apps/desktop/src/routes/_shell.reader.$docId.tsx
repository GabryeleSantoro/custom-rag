import { createFileRoute } from "@tanstack/react-router";

import { ReaderView } from "@/features/reader/reader-view";

type ReaderSearch = { page?: number; highlight?: string };

export const Route = createFileRoute("/_shell/reader/$docId")({
  validateSearch: (search: Record<string, unknown>): ReaderSearch => ({
    page: search.page ? Number(search.page) : undefined,
    highlight: typeof search.highlight === "string" ? search.highlight : undefined,
  }),
  component: ReaderView,
});
