import { useState } from "react";
import { Link, useNavigate, useParams } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ChevronDownIcon,
  ChevronRightIcon,
  FolderIcon,
  FolderPlusIcon,
  MessageSquarePlusIcon,
  PinIcon,
  PinOffIcon,
  PlusIcon,
  Trash2Icon,
} from "lucide-react";
import { toast } from "sonner";

import { ContextSidebar, SidebarSectionLabel } from "@/components/shell/context-sidebar";
import { Button } from "@/components/ui/button";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { api, type ChatProject, type ChatSession } from "@/lib/ipc";
import { relativeTime } from "@/lib/format";
import { keys, projectsQuery, sessionsQuery } from "@/lib/queries";
import { cn } from "@/lib/utils";

function sortSessions(sessions: ChatSession[]) {
  return [...sessions].sort(
    (left, right) =>
      Number(right.pinned) - Number(left.pinned) ||
      right.updated_at.localeCompare(left.updated_at),
  );
}

function SessionRow({
  session,
  projects,
  active,
  onDelete,
  onTogglePin,
  onMove,
}: {
  session: ChatSession;
  projects: ChatProject[];
  active: boolean;
  onDelete: (id: string) => void;
  onTogglePin: (session: ChatSession) => void;
  onMove: (session: ChatSession, projectId: string | null) => void;
}) {
  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>
        <div
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
            <div className="flex min-w-0 items-center gap-1.5">
              {session.pinned ? <PinIcon className="size-3 shrink-0 text-primary" /> : null}
              <p
                className={cn(
                  "truncate text-[0.8125rem] leading-tight",
                  active ? "font-medium" : "text-sidebar-foreground",
                )}
              >
                {session.title}
              </p>
            </div>
            <p className="mt-0.5 truncate text-[0.6875rem] text-muted-foreground">
              {session.message_count} message{session.message_count === 1 ? "" : "s"} ·{" "}
              {relativeTime(session.updated_at)}
            </p>
          </Link>

          <Button
            variant="ghost"
            size="icon"
            aria-label={`Delete ${session.title}`}
            onClick={() => onDelete(session.id)}
            className="absolute top-1.5 right-1 size-6 opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
          >
            <Trash2Icon className="size-3.5" />
          </Button>
        </div>
      </ContextMenuTrigger>
      <ContextMenuContent>
        <ContextMenuItem onSelect={() => onTogglePin(session)}>
          {session.pinned ? <PinOffIcon /> : <PinIcon />}
          {session.pinned ? "Unpin chat" : "Pin chat"}
        </ContextMenuItem>
        {session.project_id ? (
          <ContextMenuItem onSelect={() => onMove(session, null)}>
            <FolderIcon />
            Remove from project
          </ContextMenuItem>
        ) : null}
        {projects
          .filter((project) => project.id !== session.project_id)
          .map((project) => (
            <ContextMenuItem key={project.id} onSelect={() => onMove(session, project.id)}>
              <FolderIcon />
              Move to {project.name}
            </ContextMenuItem>
          ))}
        <ContextMenuItem variant="destructive" onSelect={() => onDelete(session.id)}>
          <Trash2Icon />
          Delete chat
        </ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  );
}

function ProjectRow({
  project,
  collapsed,
  chatCount,
  onToggle,
  onNewChat,
  onTogglePin,
  onDelete,
}: {
  project: ChatProject;
  collapsed: boolean;
  chatCount: number;
  onToggle: () => void;
  onNewChat: () => void;
  onTogglePin: () => void;
  onDelete: () => void;
}) {
  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>
        <div className="group flex items-center rounded-md hover:bg-sidebar-accent/60">
          <button
            type="button"
            className="flex min-w-0 flex-1 items-center gap-1.5 px-1.5 py-1.5 text-left"
            onClick={onToggle}
            aria-expanded={!collapsed}
          >
            {collapsed ? (
              <ChevronRightIcon className="size-3.5 shrink-0 text-muted-foreground" />
            ) : (
              <ChevronDownIcon className="size-3.5 shrink-0 text-muted-foreground" />
            )}
            <FolderIcon className="size-3.5 shrink-0 text-muted-foreground" />
            <span className="truncate text-[0.8125rem] font-medium">{project.name}</span>
            {project.pinned ? <PinIcon className="size-3 shrink-0 text-primary" /> : null}
            <span className="ml-auto shrink-0 text-[0.6875rem] text-muted-foreground">
              {chatCount}
            </span>
          </button>
          <Button
            variant="ghost"
            size="icon"
            className="mr-0.5 size-6 opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
            aria-label={`New chat in ${project.name}`}
            onClick={(event) => {
              event.stopPropagation();
              onNewChat();
            }}
          >
            <PlusIcon className="size-3.5" />
          </Button>
        </div>
      </ContextMenuTrigger>
      <ContextMenuContent>
        <ContextMenuItem onSelect={onTogglePin}>
          {project.pinned ? <PinOffIcon /> : <PinIcon />}
          {project.pinned ? "Unpin project" : "Pin project"}
        </ContextMenuItem>
        <ContextMenuItem variant="destructive" onSelect={onDelete}>
          <Trash2Icon />
          Delete project
        </ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  );
}

export function ChatSessionSidebar() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const params = useParams({ strict: false }) as { sessionId?: string };
  const sessions = useQuery(sessionsQuery);
  const projects = useQuery(projectsQuery);
  const [projectDialogOpen, setProjectDialogOpen] = useState(false);
  const [projectName, setProjectName] = useState("");
  const [collapsedProjects, setCollapsedProjects] = useState<Set<string>>(new Set());

  const invalidateChatNavigation = () => {
    void queryClient.invalidateQueries({ queryKey: keys.sessions });
    void queryClient.invalidateQueries({ queryKey: keys.projects });
  };

  const remove = useMutation({
    mutationFn: (id: string) => api.deleteSession(id),
    onSuccess: (_result, id) => {
      invalidateChatNavigation();
      if (params.sessionId === id) void navigate({ to: "/chat" });
    },
    onError: (error: Error) => toast.error("Could not delete the chat", { description: error.message }),
  });

  const createProject = useMutation({
    mutationFn: () => api.createProject({ name: projectName.trim() }),
    onSuccess: (project) => {
      void queryClient.invalidateQueries({ queryKey: keys.projects });
      setCollapsedProjects((current) => {
        const next = new Set(current);
        next.delete(project.id);
        return next;
      });
      setProjectName("");
      setProjectDialogOpen(false);
      toast.success("Project created", { description: project.name });
    },
    onError: (error: Error) => toast.error("Could not create the project", { description: error.message }),
  });

  const updateProject = useMutation({
    mutationFn: ({ id, pinned }: { id: string; pinned: boolean }) =>
      api.updateProject(id, { pinned }),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: keys.projects }),
    onError: (error: Error) => toast.error("Could not update the project", { description: error.message }),
  });

  const removeProject = useMutation({
    mutationFn: (id: string) => api.deleteProject(id),
    onSuccess: () => invalidateChatNavigation(),
    onError: (error: Error) => toast.error("Could not delete the project", { description: error.message }),
  });

  const createProjectChat = useMutation({
    mutationFn: (projectId: string) => api.createSession({ project_id: projectId }),
    onSuccess: (session) => {
      invalidateChatNavigation();
      void navigate({ to: "/chat/$sessionId", params: { sessionId: session.id } });
    },
    onError: (error: Error) => toast.error("Could not create the chat", { description: error.message }),
  });

  const updateChat = useMutation({
    mutationFn: ({ id, payload }: { id: string; payload: Parameters<typeof api.updateSession>[1] }) =>
      api.updateSession(id, payload),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: keys.sessions }),
    onError: (error: Error) => toast.error("Could not update the chat", { description: error.message }),
  });

  const allSessions = sessions.data ?? [];
  const allProjects = projects.data ?? [];
  const orderedProjects = [...allProjects].sort(
    (left, right) =>
      Number(right.pinned) - Number(left.pinned) || left.name.localeCompare(right.name),
  );
  const unassignedSessions = sortSessions(allSessions.filter((session) => !session.project_id));

  const toggleProject = (id: string) => {
    setCollapsedProjects((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <>
      <Dialog open={projectDialogOpen} onOpenChange={setProjectDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <form
            onSubmit={(event) => {
              event.preventDefault();
              if (projectName.trim()) createProject.mutate();
            }}
          >
            <DialogHeader>
              <DialogTitle>New project</DialogTitle>
              <DialogDescription>
                Keep related chats together so they are easier to find and pin as a group.
              </DialogDescription>
            </DialogHeader>
            <div className="py-4">
              <Label htmlFor="project-name">Project name</Label>
              <Input
                id="project-name"
                className="mt-1.5"
                autoFocus
                value={projectName}
                placeholder="Research notes"
                onChange={(event) => setProjectName(event.target.value)}
              />
            </div>
            <DialogFooter>
              <Button type="button" variant="ghost" onClick={() => setProjectDialogOpen(false)}>
                Cancel
              </Button>
              <Button type="submit" disabled={!projectName.trim() || createProject.isPending}>
                {createProject.isPending ? "Creating…" : "Create project"}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      <ContextSidebar
        title="Chats"
        action={
          <div className="flex items-center gap-0.5">
            <Button variant="ghost" size="icon" className="size-7" asChild>
              <Link to="/chat" aria-label="New chat">
                <MessageSquarePlusIcon className="size-4" />
              </Link>
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="size-7"
              aria-label="New project"
              onClick={() => setProjectDialogOpen(true)}
            >
              <FolderPlusIcon className="size-4" />
            </Button>
          </div>
        }
      >
        {sessions.isLoading || projects.isLoading ? (
          <div className="space-y-1 px-1 pt-2">
            <Skeleton className="h-11 w-full" />
            <Skeleton className="h-11 w-full" />
          </div>
        ) : null}

        {orderedProjects.length ? <SidebarSectionLabel>Projects</SidebarSectionLabel> : null}
        <nav className="flex flex-col gap-0.5">
          {orderedProjects.map((project) => {
            const projectSessions = sortSessions(
              allSessions.filter((session) => session.project_id === project.id),
            );
            const collapsed = collapsedProjects.has(project.id);
            return (
              <div key={project.id}>
                <ProjectRow
                  project={project}
                  collapsed={collapsed}
                  chatCount={projectSessions.length}
                  onToggle={() => toggleProject(project.id)}
                  onNewChat={() => createProjectChat.mutate(project.id)}
                  onTogglePin={() => updateProject.mutate({ id: project.id, pinned: !project.pinned })}
                  onDelete={() => removeProject.mutate(project.id)}
                />
                {!collapsed ? (
                  <div className="ml-3 border-l border-sidebar-border pl-1">
                    {projectSessions.map((session) => (
                      <SessionRow
                        key={session.id}
                        session={session}
                        projects={allProjects}
                        active={params.sessionId === session.id}
                        onDelete={(id) => remove.mutate(id)}
                        onTogglePin={(chat) =>
                          updateChat.mutate({ id: chat.id, payload: { pinned: !chat.pinned } })
                        }
                        onMove={(chat, projectId) =>
                          updateChat.mutate({ id: chat.id, payload: { project_id: projectId } })
                        }
                      />
                    ))}
                  </div>
                ) : null}
              </div>
            );
          })}
        </nav>

        {unassignedSessions.length ? <SidebarSectionLabel>Chats</SidebarSectionLabel> : null}
        <nav className="flex flex-col gap-0.5">
          {unassignedSessions.map((session) => (
            <SessionRow
              key={session.id}
              session={session}
              projects={allProjects}
              active={params.sessionId === session.id}
              onDelete={(id) => remove.mutate(id)}
              onTogglePin={(chat) =>
                updateChat.mutate({ id: chat.id, payload: { pinned: !chat.pinned } })
              }
              onMove={(chat, projectId) =>
                updateChat.mutate({ id: chat.id, payload: { project_id: projectId } })
              }
            />
          ))}
        </nav>

        {!sessions.isLoading && !projects.isLoading && !allSessions.length && !allProjects.length ? (
          <p className="px-2 py-6 text-center text-xs text-muted-foreground">
            No chats yet. Ask something to start one.
          </p>
        ) : null}
      </ContextSidebar>
    </>
  );
}
