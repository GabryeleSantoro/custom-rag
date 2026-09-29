import { describe, expect, it } from "vitest";

import { levelOf } from "./log-pane";

describe("levelOf", () => {
  it("reads the level from the text, whatever stream the line came from", () => {
    expect(levelOf("2026-09-29 08:15:02Z [err] ERROR ragcore.llm: HTTP 502: bad gateway")).toBe("error");
    expect(levelOf("[shell] stream POST /conversions/slides broken after 300s: stream broken")).toBe("error");
    expect(levelOf("[err] WARNING ragcore.library: source folder missing")).toBe("warn");
    expect(levelOf("[err] INFO ragcore.llm: finished in 12.0s, finish reason stop")).toBe("info");
    expect(levelOf("[err] INFO ragcore.conversions: conversion done: 1 saved, 0 not converted")).toBe("info");
  });
});
