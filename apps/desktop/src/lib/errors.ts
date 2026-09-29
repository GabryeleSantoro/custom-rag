import type { TFunction } from "i18next";

import i18n from "@/lib/i18n";
import { IpcError } from "@/lib/ipc";

export { parseIpcError } from "@/lib/ipc";

type Translate = TFunction | ((key: string, options?: Record<string, unknown>) => string);

/**
 * User-facing text for any error. A code the catalog knows is translated; anything
 * else (an old-style string, a brand-new code) shows its English message as is.
 */
export function errorText(
  error: unknown,
  t: Translate = i18n.t.bind(i18n),
  exists: (key: string) => boolean = (key) => i18n.exists(key),
): string {
  if (error instanceof IpcError && error.code && exists(`errors.${error.code}`)) {
    return (t as (key: string, options?: Record<string, unknown>) => string)(
      `errors.${error.code}`,
      error.params ?? {},
    );
  }
  return error instanceof Error ? error.message : String(error);
}
