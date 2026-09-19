import { useEffect } from "react";
import { Outlet, createRootRoute } from "@tanstack/react-router";

import { useAutoUpdate } from "@/features/settings/updates";
import { JobsProvider } from "@/lib/jobs-context";
import { Toaster } from "@/components/ui/sonner";
import { installWindowDragHandler } from "@/lib/window-drag";

export const Route = createRootRoute({
  component: RootLayout,
});

function RootLayout() {
  useEffect(() => installWindowDragHandler(), []);
  useAutoUpdate();

  return (
    <JobsProvider>
      <div className="h-full">
        <Outlet />
        <Toaster position="bottom-right" />
      </div>
    </JobsProvider>
  );
}
