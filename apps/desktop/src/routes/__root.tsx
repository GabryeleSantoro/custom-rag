import { Outlet, createRootRoute } from "@tanstack/react-router";

import { JobsProvider } from "@/lib/jobs-context";
import { Toaster } from "@/components/ui/sonner";

export const Route = createRootRoute({
  component: RootLayout,
});

function RootLayout() {
  return (
    <JobsProvider>
      <div className="h-full">
        <Outlet />
        <Toaster position="bottom-right" />
      </div>
    </JobsProvider>
  );
}
