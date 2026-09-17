import { ConstructionIcon } from "lucide-react";

/** Temporary body for screens that are scaffolded but not yet built out. */
export function Placeholder({ name, note }: { name: string; note?: string }) {
  return (
    <div className="grid h-full place-items-center p-10 text-center">
      <div className="max-w-sm space-y-2">
        <ConstructionIcon className="mx-auto size-6 text-muted-foreground" strokeWidth={1.5} />
        <p className="text-sm font-medium">{name}</p>
        {note ? <p className="text-xs text-muted-foreground">{note}</p> : null}
      </div>
    </div>
  );
}
