import { describe, expect, it } from "vitest";
import { resolveLanguage } from "./i18n";

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
