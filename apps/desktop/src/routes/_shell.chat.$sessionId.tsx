import { createFileRoute } from "@tanstack/react-router";

import { ChatView } from "@/features/chat/chat-view";

export const Route = createFileRoute("/_shell/chat/$sessionId")({
  component: ChatSessionRoute,
});

function ChatSessionRoute() {
  const { sessionId } = Route.useParams();
  return <ChatView sessionId={sessionId} />;
}
