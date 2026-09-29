import { describe, expect, it } from "vitest";
import de from "./de.json";
import en from "./en.json";
import es from "./es.json";
import fr from "./fr.json";
import it_ from "./it.json";

type Tree = { [key: string]: string | Tree };

function flatten(tree: Tree, prefix = ""): Record<string, string> {
  return Object.entries(tree).reduce<Record<string, string>>((out, [key, value]) => {
    const path = prefix ? `${prefix}.${key}` : key;
    return typeof value === "string"
      ? { ...out, [path]: value }
      : { ...out, ...flatten(value, path) };
  }, {});
}

const placeholders = (text: string) =>
  [...text.matchAll(/{{\s*(\w+)\s*}}/g)].map((m) => m[1]).sort();
const base = flatten(en as Tree);

describe.each([
  ["it", it_],
  ["fr", fr],
  ["de", de],
  ["es", es],
] as const)("%s catalog", (_name, catalog) => {
  const flat = flatten(catalog as Tree);
  it("has exactly the keys of en", () => {
    expect(Object.keys(flat).sort()).toEqual(Object.keys(base).sort());
  });
  it("keeps the same placeholders", () => {
    for (const [key, text] of Object.entries(base)) {
      expect(placeholders(flat[key] ?? ""), key).toEqual(placeholders(text));
    }
  });
  it("has no empty strings", () => {
    for (const [key, text] of Object.entries(flat)) expect(text.trim(), key).not.toBe("");
  });
});
