/** Small formatters shared across features. */

const UNITS = ["B", "KB", "MB", "GB", "TB"];

export function bytes(value: number | null | undefined): string {
  if (!value) return "0 B";
  const exponent = Math.min(Math.floor(Math.log(value) / Math.log(1024)), UNITS.length - 1);
  const scaled = value / 1024 ** exponent;
  return `${scaled.toFixed(scaled >= 10 || exponent === 0 ? 0 : 1)} ${UNITS[exponent]}`;
}

export function ms(value: number | null | undefined): string {
  if (value == null) return "—";
  if (value < 1) return "<1 ms";
  if (value < 1000) return `${Math.round(value)} ms`;
  return `${(value / 1000).toFixed(value < 10_000 ? 2 : 1)} s`;
}

export function count(value: number): string {
  if (value < 1000) return String(value);
  if (value < 1_000_000) return `${(value / 1000).toFixed(value < 10_000 ? 1 : 0)}k`;
  return `${(value / 1_000_000).toFixed(1)}M`;
}

const RELATIVE = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
const STEPS: [Intl.RelativeTimeFormatUnit, number][] = [
  ["second", 60],
  ["minute", 60],
  ["hour", 24],
  ["day", 7],
  ["week", 4.35],
  ["month", 12],
  ["year", Infinity],
];

export function relativeTime(iso: string | null | undefined): string {
  if (!iso) return "never";
  let delta = (new Date(iso).getTime() - Date.now()) / 1000;
  for (const [unit, span] of STEPS) {
    if (Math.abs(delta) < span) return RELATIVE.format(Math.round(delta), unit);
    delta /= span;
  }
  return "long ago";
}

export function shortPath(path: string, segments = 2): string {
  const parts = path.split("/").filter(Boolean);
  return parts.length <= segments ? path : `…/${parts.slice(-segments).join("/")}`;
}

/** Process uptime. Coarse on purpose — a sidecar that has been up 3 h 04 m is just "up". */
export function duration(seconds: number | null | undefined): string {
  if (seconds == null || seconds <= 0) return "—";
  if (seconds < 60) return `${Math.floor(seconds)}s`;
  const total = Math.floor(seconds);
  const minutes = Math.floor(total / 60) % 60;
  const hours = Math.floor(total / 3600);
  if (hours === 0) return `${minutes}m ${String(total % 60).padStart(2, "0")}s`;
  return `${hours}h ${String(minutes).padStart(2, "0")}m`;
}
