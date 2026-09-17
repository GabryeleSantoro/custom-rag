import { useEffect, useRef, useState } from "react";
import { ArrowUpIcon, SquareIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const MAX_ROWS_PX = 180;

export function Composer({
  onSend,
  onStop,
  streaming,
  disabled,
  placeholder = "Ask about your documents…",
  children,
}: {
  onSend: (text: string) => void;
  onStop: () => void;
  streaming: boolean;
  disabled?: boolean;
  placeholder?: string;
  /** Filter chips and the mode selector sit above the input. */
  children?: React.ReactNode;
}) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, MAX_ROWS_PX)}px`;
  }, [value]);

  const submit = () => {
    const text = value.trim();
    if (!text || streaming || disabled) return;
    onSend(text);
    setValue("");
  };

  return (
    <div className="border-t border-border bg-background px-5 py-3">
      <div className="mx-auto w-full max-w-3xl">
        {children ? <div className="mb-2 flex flex-wrap gap-1.5">{children}</div> : null}

        <div
          className={cn(
            "flex items-end gap-2 rounded-xl border border-input bg-card p-2 transition-colors",
            "focus-within:border-ring focus-within:ring-[3px] focus-within:ring-ring/25",
            disabled && "opacity-60",
          )}
        >
          <textarea
            ref={ref}
            rows={1}
            value={value}
            disabled={disabled}
            placeholder={placeholder}
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submit();
              }
            }}
            className="max-h-45 flex-1 resize-none bg-transparent px-2 py-1.5 text-sm outline-none placeholder:text-muted-foreground"
          />
          {streaming ? (
            <Button size="icon" variant="secondary" className="size-8" onClick={onStop}>
              <SquareIcon className="size-3.5 fill-current" />
              <span className="sr-only">Stop</span>
            </Button>
          ) : (
            <Button
              size="icon"
              className="size-8"
              disabled={!value.trim() || disabled}
              onClick={submit}
            >
              <ArrowUpIcon className="size-4" />
              <span className="sr-only">Send</span>
            </Button>
          )}
        </div>

        <p className="mt-1.5 px-1 text-[0.6875rem] text-muted-foreground">
          Enter sends, Shift+Enter adds a line.
        </p>
      </div>
    </div>
  );
}
