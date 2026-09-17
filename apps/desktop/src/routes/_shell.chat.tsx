import { Outlet, createFileRoute } from "@tanstack/react-router";

import { ChatSessionSidebar } from "@/features/chat/session-sidebar";

export const Route = createFileRoute("/_shell/chat")({
  component: ChatSection,
});

function ChatSection() {
  return (
    <>
      <ChatSessionSidebar />
      <Outlet />
    </>
  );
}
