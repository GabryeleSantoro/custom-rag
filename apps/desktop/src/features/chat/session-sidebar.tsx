import { Link, useNavigate, useParams } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { MessageSquarePlusIcon, Trash2Icon } from "lucide-react";
import { toast } from "sonner";

import { ContextSidebar } from "@/components/shell/context-sidebar";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/ipc";
import { relativeTime } from "@/lib/format";
import { keys, sessionsQuery } from "@/lib/queries";
import { cn } from "@/lib/utils";

export function ChatSessionSidebar() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const params = useParams({ strict: false }) as { sessionId?: string };
  const sessions = useQuery(sessionsQuery);

  const remove = useMutation({
    mutationFn: (id: string) => api.deleteSession(id),
    onSuccess: (_result, id) => {
      void queryClient.invalidateQueries({ queryKey: keys.sessions });
      if (params.sessionId === id) void navigate({ to: "/chat" });
    },
    onError: (error: Error) => toast.error("Could not delete the chat", { description: error.message }),
  });

  return (
    <ContextSidebar
      title="Chats"
      action={
        <Button variant="ghost" size="icon" className="size-7" asChild>
          <Link to="/chat" aria-label="New chat">
            <MessageSquarePlusIcon className="size-4" />
          </Link>
        </Button>
      }
    >
      {sessions.isLoading ? (
        <div className="space-y-1 px-1 pt-2">
          <Skeleton className="h-11 w-full" />
          <Skeleton className="h-11 w-full" />
        </div>
      ) : null}

      {sessions.data?.length === 0 ? (
        <p className="px-2 py-6 text-center text-xs text-muted-foreground">
          No chats yet. Ask something to start one.
        </p>
      ) : null}

      <nav className="flex flex-col gap-0.5 pt-1">
        {(sessions.data ?? []).map((session) => {
          const active = params.sessionId === session.id;
          return (
            <div
              key={session.id}
              className={cn(
                "group relative rounded-md transition-colors",
                active ? "bg-sidebar-accent" : "hover:bg-sidebar-accent/60",
              )}
            >
              <Link
                to="/chat/$sessionId"
                params={{ sessionId: session.id }}
                className="block px-2 py-1.5 pr-8"
              >
                <p
                  className={cn(
                    "truncate text-[0.8125rem] leading-tight",
                    active ? "font-medium" : "text-sidebar-foreground",
                  )}
                >
                  {session.title}
                </p>
                <p className="mt-0.5 truncate text-[0.6875rem] text-muted-foreground">
                  {session.message_count} message{session.message_count === 1 ? "" : "s"} ·{" "}
                  {relativeTime(session.updated_at)}
                </p>
              </Link>

              <Button
                variant="ghost"
                size="icon"
                aria-label={`Delete ${session.title}`}
                onClick={() => remove.mutate(session.id)}
                className="absolute top-1.5 right-1 size-6 opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
              >
                <Trash2Icon className="size-3.5" />
              </Button>
            </div>
          );
        })}
      </nav>
    </ContextSidebar>
  );
}
