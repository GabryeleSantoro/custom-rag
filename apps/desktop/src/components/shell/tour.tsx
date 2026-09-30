import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";

const SEEN_KEY = "ibid.tour";

/** `target` matches a `data-tour` attribute; no target = centered card, full dim. */
const STEPS = [
  { id: "welcome" },
  { id: "chat", target: "/chat" },
  { id: "library", target: "/library" },
  { id: "convert", target: "/convert" },
  { id: "models", target: "/models" },
  { id: "settings", target: "/settings" },
  { id: "diagnostics", target: "/diagnostics" },
  { id: "done" },
] as const;

const PAD = 6;

const seen = () => {
  try {
    return localStorage.getItem(SEEN_KEY) === "done";
  } catch {
    return false;
  }
};

/**
 * One-time guided first open: dims the app and lights up each rail entry in
 * turn. The "light" is a single box with a huge outer shadow, so the cutout
 * is pure CSS and animates between targets with a plain transition.
 */
export function Tour() {
  const { t } = useTranslation();
  const [open, setOpen] = useState(() => !seen());
  const [index, setIndex] = useState(0);
  const [rect, setRect] = useState<DOMRect | null>(null);
  const nextRef = useRef<HTMLButtonElement>(null);
  const step = STEPS[index];
  const target = "target" in step ? step.target : null;
  const last = index === STEPS.length - 1;

  useLayoutEffect(() => {
    if (!open) return;
    const measure = () =>
      setRect(target ? (document.querySelector(`[data-tour="${target}"]`)?.getBoundingClientRect() ?? null) : null);
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [open, target]);

  const close = () => {
    try {
      localStorage.setItem(SEEN_KEY, "done");
    } catch {
      // Private storage unavailable: the tour just shows again next launch.
    }
    setOpen(false);
  };
  const next = () => (last ? close() : setIndex((i) => i + 1));
  const back = () => setIndex((i) => Math.max(i - 1, 0));

  useEffect(() => {
    if (!open) return;
    nextRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
      else if (e.key === "ArrowRight") next();
      else if (e.key === "ArrowLeft") back();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  if (!open) return null;

  const dim = "0 0 0 9999px rgb(0 0 0 / 0.62)";
  const glow = "0 0 0 2px var(--primary), 0 0 28px 8px color-mix(in oklch, var(--primary) 55%, transparent)";

  return (
    <div className="fixed inset-0 z-100" role="dialog" aria-modal="true" aria-labelledby="tour-title">
      {/* The light: shrinks to nothing at screen centre when there is no target. */}
      <div
        aria-hidden
        className="pointer-events-none fixed rounded-xl transition-all duration-500 ease-[cubic-bezier(0.22,1,0.36,1)]"
        style={
          rect
            ? {
                left: rect.left - PAD,
                top: rect.top - PAD,
                width: rect.width + PAD * 2,
                height: rect.height + PAD * 2,
                boxShadow: `${dim}, ${glow}`,
              }
            : { left: "50%", top: "50%", width: 0, height: 0, boxShadow: dim }
        }
      />

      <div
        key={step.id}
        className="page-enter fixed w-80 rounded-xl border border-border bg-popover p-4 text-popover-foreground shadow-2xl"
        style={
          rect
            ? { left: rect.right + PAD + 16, top: `clamp(16px, ${rect.top - 12}px, calc(100vh - 220px))` }
            : { left: "50%", top: "50%", translate: "-50% -50%" }
        }
      >
        <p className="text-[0.6875rem] tabular-nums text-muted-foreground">
          {index + 1} / {STEPS.length}
        </p>
        <h2 id="tour-title" className="mt-1 text-sm font-semibold tracking-tight">
          {t(`tour.${step.id}.title`)}
        </h2>
        <p className="mt-1.5 text-[0.8125rem] leading-[1.55] text-muted-foreground">
          {t(`tour.${step.id}.body`)}
        </p>
        <div className="mt-4 flex items-center justify-between gap-2">
          <Button variant="ghost" size="sm" className="h-8" onClick={close}>
            {t("tour.skip")}
          </Button>
          <div className="flex gap-2">
            {index > 0 ? (
              <Button variant="outline" size="sm" className="h-8" onClick={back}>
                {t("common.back")}
              </Button>
            ) : null}
            <Button ref={nextRef} size="sm" className="h-8" onClick={next}>
              {last ? t("tour.finish") : t("tour.next")}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
