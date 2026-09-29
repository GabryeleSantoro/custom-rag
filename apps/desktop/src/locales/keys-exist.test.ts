import { describe, expect, it } from "vitest";
import en from "./en.json";

const sources = import.meta.glob<string>(["../**/*.ts", "../**/*.tsx", "!../**/*.test.ts"], {
  query: "?raw",
  import: "default",
  eager: true,
});

type Tree = { [key: string]: string | Tree };

function has(tree: Tree, key: string): boolean {
  const node = key.split(".").reduce<string | Tree | undefined>(
    (current, part) => (typeof current === "object" ? current[part] : undefined),
    tree,
  );
  return node !== undefined;
}

// t("a.b") and i18n.t("a.b"): only static keys; template-literal keys are built from
// enums (roles, statuses) and covered by the catalog parity test instead.
const KEY = /\bt\(\s*"([a-zA-Z0-9_.]+)"/g;

describe("translation keys used in source", () => {
  it("all exist in the English catalog", () => {
    const missing: string[] = [];
    for (const [file, source] of Object.entries(sources)) {
      for (const [, key] of source.matchAll(KEY)) {
        const plural = has(en as Tree, `${key}_other`);
        if (!has(en as Tree, key) && !plural) missing.push(`${file}: ${key}`);
      }
    }
    expect(missing).toEqual([]);
  });
});
