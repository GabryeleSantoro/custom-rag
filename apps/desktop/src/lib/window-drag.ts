import { isTauri } from "@tauri-apps/api/core";
import { getCurrentWindow } from "@tauri-apps/api/window";

const interactiveSelector =
  ".no-drag, [data-tauri-drag-region=\"false\"], button, a, input, textarea, select, [role=\"button\"]";

/**
 * The CSS drag region is not handled consistently by every webview/input
 * combination. Keep it as the native fast path and explicitly start a drag
 * for mouse users as a fallback.
 */
export function installWindowDragHandler(): () => void {
  if (!isTauri()) return () => undefined;

  const window = getCurrentWindow();
  const onMouseDown = (event: MouseEvent) => {
    if (event.button !== 0) return;

    const target = event.target;
    if (!(target instanceof Element)) return;

    const dragRegion = target.closest(".drag-region");
    if (!dragRegion || target.closest(interactiveSelector)) return;

    void window.startDragging();
  };

  document.addEventListener("mousedown", onMouseDown);
  return () => document.removeEventListener("mousedown", onMouseDown);
}
