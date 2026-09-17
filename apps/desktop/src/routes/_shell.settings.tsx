import { Outlet, createFileRoute } from "@tanstack/react-router";

import { SettingsSidebar } from "@/features/settings/settings-sidebar";

export const Route = createFileRoute("/_shell/settings")({
  component: SettingsSection,
});

function SettingsSection() {
  return (
    <>
      <SettingsSidebar />
      <Outlet />
    </>
  );
}
