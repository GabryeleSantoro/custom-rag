import { useEffect } from "react";
import { Outlet, createFileRoute, useNavigate, useRouterState } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";

import { IconRail } from "@/components/shell/icon-rail";
import { Tour } from "@/components/shell/tour";
import { settingsQuery } from "@/lib/queries";

export const Route = createFileRoute("/_shell")({
  component: Shell,
});

function Shell() {
  const navigate = useNavigate();
  const settings = useQuery(settingsQuery);
  const onboarded = settings.data?.onboarded;
  // Re-key per section (not per param) so /chat/:id switches don't replay it.
  const section = useRouterState({ select: (s) => s.location.pathname.split("/")[1] ?? "" });

  // Deliberately not a `beforeLoad` redirect: the core may still be starting
  // when the window opens, and a failed settings fetch must not strand the
  // user outside Diagnostics, which is where they would go to find out why.
  useEffect(() => {
    if (onboarded === false) void navigate({ to: "/onboarding", replace: true });
  }, [onboarded, navigate]);

  return (
    <div className="flex h-full overflow-hidden">
      <IconRail devMode={import.meta.env.DEV} />
      <div key={section} className="page-enter flex min-w-0 flex-1">
        <Outlet />
      </div>
      {onboarded ? <Tour /> : null}
    </div>
  );
}
