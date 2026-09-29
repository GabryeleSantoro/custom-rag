import { afterEach, describe, expect, it, vi } from "vitest";
import { currentLanguage, resolveLanguage, setSystemLocale } from "./i18n";

describe("resolveLanguage", () => {
  it("system follows the OS base language", () => {
    expect(resolveLanguage("system", "it-IT")).toBe("it");
    expect(resolveLanguage("system", "de-AT")).toBe("de");
    expect(resolveLanguage("system", "fr_CA")).toBe("fr");
    expect(resolveLanguage("system", "ES")).toBe("es");
  });
  it("falls back to en for unsupported or missing OS language", () => {
    expect(resolveLanguage("system", "pt-BR")).toBe("en");
    expect(resolveLanguage("system", "")).toBe("en");
    expect(resolveLanguage("system", undefined)).toBe("en");
  });
  it("an explicit choice wins over the OS", () => {
    expect(resolveLanguage("fr", "it-IT")).toBe("fr");
  });
  it("garbage preferences behave like system", () => {
    expect(resolveLanguage("klingon", "it-IT")).toBe("it");
    expect(resolveLanguage(null, "de")).toBe("de");
  });
});

describe("system locale from the shell", () => {
  afterEach(() => {
    setSystemLocale(undefined);
    vi.unstubAllGlobals();
  });

  it("wins over the webview language when the preference is system", () => {
    vi.stubGlobal("localStorage", { getItem: () => null, setItem: () => {}, removeItem: () => {} });
    vi.stubGlobal("navigator", { language: "en-US" });

    setSystemLocale("it-IT");

    expect(currentLanguage()).toBe("it");
  });

  it("does not override an explicit choice", () => {
    vi.stubGlobal("localStorage", { getItem: () => "fr", setItem: () => {}, removeItem: () => {} });

    setSystemLocale("it-IT");

    expect(currentLanguage()).toBe("fr");
  });
});
