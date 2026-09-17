/**
 * Platform facts the layout needs before any async Tauri call resolves.
 * Derived from the webview UA so it is available on the first render — the
 * traffic-light gutter must not pop in after paint.
 */
const ua = typeof navigator === "undefined" ? "" : navigator.userAgent;

export const isMac = /Mac(intosh| OS X)/.test(ua);
export const isWindows = /Windows/.test(ua);
export const isLinux = !isMac && !isWindows;

/**
 * macOS keeps its native traffic lights floating over our chrome
 * (tauri.conf.json: titleBarStyle "Overlay"), so the rail owes them a gutter.
 */
export const trafficLightGutter = isMac ? 28 : 0;
