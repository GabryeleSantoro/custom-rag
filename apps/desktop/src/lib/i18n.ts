import i18n from "i18next";
import { isTauri } from "@tauri-apps/api/core";
import { initReactI18next } from "react-i18next";

import de from "@/locales/de.json";
import en from "@/locales/en.json";
import es from "@/locales/es.json";
import fr from "@/locales/fr.json";
import it from "@/locales/it.json";

export const LANGUAGES = ["en", "it", "fr", "de", "es"] as const;
export type Language = (typeof LANGUAGES)[number];

/** Each language in its own name: a picker must stay readable in a language you cannot read. */
export const LANGUAGE_NAMES: Record<Language, string> = {
  en: "English",
  it: "Italiano",
  fr: "Français",
  de: "Deutsch",
  es: "Español",
};
export type LanguagePref = "system" | Language;

const STORAGE_KEY = "language";

const isLanguage = (value: unknown): value is Language =>
  typeof value === "string" && (LANGUAGES as readonly string[]).includes(value);

export function resolveLanguage(pref: unknown, navigatorLang: string | undefined): Language {
  if (isLanguage(pref)) return pref;
  const base = (navigatorLang ?? "").split(/[-_]/)[0].toLowerCase();
  return isLanguage(base) ? base : "en";
}

export function getLanguagePref(): LanguagePref {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return isLanguage(stored) ? stored : "system";
  } catch {
    return "system";
  }
}

// The webview's own language follows the app bundle's localizations on macOS, so it
// can say "en" on an Italian Mac. The shell reports the real OS locale instead.
let systemLocale: string | undefined;

export function currentLanguage(): Language {
  return resolveLanguage(getLanguagePref(), systemLocale ?? globalThis.navigator?.language);
}

export function setSystemLocale(locale: string | undefined): void {
  systemLocale = locale;
  if (typeof document !== "undefined") document.documentElement.lang = currentLanguage();
  void i18n.changeLanguage(currentLanguage());
}

/** Asks the Rust shell for the OS locale. A plain browser keeps `navigator.language`. */
export async function detectSystemLocale(): Promise<void> {
  if (!isTauri()) return;
  try {
    const { locale } = await import("@tauri-apps/plugin-os");
    setSystemLocale((await locale()) ?? undefined);
  } catch {
    // Permission missing or plugin absent: the webview language stays in charge.
  }
}

export function setLanguagePref(pref: LanguagePref): void {
  try {
    if (pref === "system") localStorage.removeItem(STORAGE_KEY);
    else localStorage.setItem(STORAGE_KEY, pref);
  } catch {
    // Private mode: the choice lasts until reload.
  }
  const language = currentLanguage();
  document.documentElement.lang = language;
  void i18n.changeLanguage(language);
}

void i18n.use(initReactI18next).init({
  resources: {
    en: { translation: en },
    it: { translation: it },
    fr: { translation: fr },
    de: { translation: de },
    es: { translation: es },
  },
  lng: currentLanguage(),
  fallbackLng: "en",
  interpolation: { escapeValue: false }, // React already escapes.
  returnNull: false,
});
if (typeof document !== "undefined") document.documentElement.lang = currentLanguage();

export default i18n;
