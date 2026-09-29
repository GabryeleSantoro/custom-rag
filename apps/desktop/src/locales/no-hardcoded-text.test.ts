import { describe, expect, it } from "vitest";

// Dev-only surface, allowed to stay English.
const ALLOW = ["features/eval/", "routes/_shell.eval.tsx", "routeTree.gen.ts", "components/ui/"];

const sources = import.meta.glob<string>("../**/*.tsx", {
  query: "?raw",
  import: "default",
  eager: true,
});

// JSX text node starting with a capitalised word: >Add source< or >\n  Add source\n
const JSX_TEXT = /(?<![=-])>[ \t\n]*([A-Z][a-z]+(?: [A-Za-z][a-z']*)*)[ \t\n]*</g;
const ATTR = /\b(?:placeholder|title|aria-label|alt|label|description|hint)="([A-Z][a-z][^"]*)"/g;
// Multi-line JSX text (paragraphs): a line of prose between tags.
const PROSE = /^[ \t]+[A-Z][a-z]+ [a-z]+ [^<>{}=;\n]*[a-z.,]$/gm;

describe("no hardcoded UI text", () => {
  for (const [file, source] of Object.entries(sources)) {
    const rel = file.replace("../", "");
    if (ALLOW.some((a) => rel.includes(a))) continue;
    it(rel, () => {
      const hits = [...source.matchAll(JSX_TEXT), ...source.matchAll(ATTR)].map((m) => m[1]);
      const prose = [...source.matchAll(PROSE)]
        .map((m) => m[0].trim())
        .filter((line) => !/^(import|export|return|const|let|if|type|\/\/|\*)/.test(line));
      expect([...hits, ...prose]).toEqual([]);
    });
  }
});
