import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { open } from "@tauri-apps/plugin-dialog";
import { FolderOpenIcon } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { api } from "@/lib/ipc";
import { keys } from "@/lib/queries";

const DEFAULT_INCLUDE = "**/*.md, **/*.txt, **/*.pdf, **/*.docx";
const DEFAULT_EXCLUDE = "**/node_modules/**, **/.git/**";

function globs(value: string): string[] {
  return value
    .split(",")
    .map((glob) => glob.trim())
    .filter(Boolean);
}

export function AddSourceDialog({ trigger }: { trigger: React.ReactNode }) {
  const queryClient = useQueryClient();
  const [openDialog, setOpenDialog] = useState(false);
  const [path, setPath] = useState("");
  const [include, setInclude] = useState(DEFAULT_INCLUDE);
  const [exclude, setExclude] = useState(DEFAULT_EXCLUDE);
  const [maxFileMb, setMaxFileMb] = useState(100);
  const [watch, setWatch] = useState(true);

  const add = useMutation({
    mutationFn: () =>
      api.addSource({
        path,
        include_globs: globs(include),
        exclude_globs: globs(exclude),
        max_file_mb: maxFileMb,
        watch,
      }),
    onSuccess: (source) => {
      void queryClient.invalidateQueries({ queryKey: keys.sources });
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
      setOpenDialog(false);
      setPath("");
      toast.success("Source added", { description: source.path });
    },
    onError: (error: Error) => toast.error("Could not add the source", { description: error.message }),
  });

  const pick = async () => {
    const selected = await open({ directory: true, multiple: false, title: "Choose a folder" });
    if (typeof selected === "string") setPath(selected);
  };

  return (
    <Dialog open={openDialog} onOpenChange={setOpenDialog}>
      <DialogTrigger asChild>{trigger}</DialogTrigger>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Add a source folder</DialogTitle>
          <DialogDescription>
            Files matching the include patterns are parsed, chunked and embedded locally. Nothing
            leaves the device during indexing.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="space-y-1.5">
            <Label htmlFor="source-path">Folder</Label>
            <div className="flex gap-2">
              <Input
                id="source-path"
                value={path}
                placeholder="/Users/you/Documents/research"
                onChange={(event) => setPath(event.target.value)}
              />
              <Button type="button" variant="secondary" onClick={pick}>
                <FolderOpenIcon className="size-4" />
                Browse
              </Button>
            </div>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label htmlFor="include">Include</Label>
              <Input id="include" value={include} onChange={(e) => setInclude(e.target.value)} />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="exclude">Exclude</Label>
              <Input id="exclude" value={exclude} onChange={(e) => setExclude(e.target.value)} />
            </div>
          </div>

          <div className="flex items-end gap-6">
            <div className="w-32 space-y-1.5">
              <Label htmlFor="max-mb">Max file size</Label>
              <div className="flex items-center gap-2">
                <Input
                  id="max-mb"
                  type="number"
                  min={1}
                  value={maxFileMb}
                  onChange={(event) => setMaxFileMb(Number(event.target.value) || 1)}
                />
                <span className="text-xs text-muted-foreground">MB</span>
              </div>
            </div>
            <div className="flex items-center gap-2 pb-2">
              <Switch id="watch" checked={watch} onCheckedChange={setWatch} />
              <Label htmlFor="watch" className="font-normal">
                Re-index when files change
              </Label>
            </div>
          </div>
        </div>

        <DialogFooter>
          <Button variant="ghost" onClick={() => setOpenDialog(false)}>
            Cancel
          </Button>
          <Button disabled={!path.trim() || add.isPending} onClick={() => add.mutate()}>
            {add.isPending ? "Adding…" : "Add source"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
