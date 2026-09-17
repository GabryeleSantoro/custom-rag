import { createFileRoute } from "@tanstack/react-router";

import { ChatView } from "@/features/chat/chat-view";

export const Route = createFileRoute("/_shell/chat/")({
  component: () => <ChatView sessionId={null} />,
});
