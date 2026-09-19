import { describe, expect, it, vi } from "vitest";

import { bytes, count, duration, ms, relativeTime, shortPath } from "./format";

describe("bytes", () => {
  it("scales to the largest unit that keeps the number small", () => {
    expect(bytes(512)).toBe("512 B");
    expect(bytes(2048)).toBe("2.0 KB");
    expect(bytes(639_000_000)).toBe("609 MB");
    expect(bytes(5 * 1024 ** 3)).toBe("5.0 GB");
  });

  it("drops the decimal once the number is big enough to carry itself", () => {
    expect(bytes(15 * 1024)).toBe("15 KB");
    expect(bytes(1023)).toBe("1023 B");
  });

  it("never invents a unit past terabytes", () => {
    expect(bytes(5 * 1024 ** 6)).toMatch(/TB$/);
  });

  it("treats nothing, zero and null alike", () => {
    expect(bytes(0)).toBe("0 B");
    expect(bytes(null)).toBe("0 B");
    expect(bytes(undefined)).toBe("0 B");
  });
});

describe("ms", () => {
  it("shows sub-millisecond latency as a floor rather than 0", () => {
    expect(ms(0.4)).toBe("<1 ms");
  });

  it("rounds milliseconds and switches to seconds past a second", () => {
    expect(ms(1)).toBe("1 ms");
    expect(ms(842.6)).toBe("843 ms");
    expect(ms(1500)).toBe("1.50 s");
    expect(ms(42_000)).toBe("42.0 s");
  });

  it("distinguishes an unmeasured value from zero", () => {
    expect(ms(null)).toBe("—");
    expect(ms(undefined)).toBe("—");
    expect(ms(0)).toBe("<1 ms");
  });
});

describe("count", () => {
  it("passes small numbers through untouched", () => {
    expect(count(0)).toBe("0");
    expect(count(999)).toBe("999");
  });

  it("abbreviates thousands and millions", () => {
    expect(count(1200)).toBe("1.2k");
    expect(count(42_000)).toBe("42k");
    expect(count(2_500_000)).toBe("2.5M");
  });
});

describe("relativeTime", () => {
  it("says never when there is no timestamp", () => {
    expect(relativeTime(null)).toBe("never");
    expect(relativeTime(undefined)).toBe("never");
    expect(relativeTime("")).toBe("never");
  });

  it("picks the coarsest unit that still fits", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-03-01T12:00:00Z"));

    expect(relativeTime("2026-03-01T11:59:30Z")).toMatch(/second/);
    expect(relativeTime("2026-03-01T11:30:00Z")).toMatch(/minute/);
    expect(relativeTime("2026-02-28T12:00:00Z")).toMatch(/yesterday|day/);
    expect(relativeTime("2023-03-01T12:00:00Z")).toMatch(/year/);

    vi.useRealTimers();
  });

  it("handles a future timestamp without going negative in the wrong direction", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-03-01T12:00:00Z"));

    expect(relativeTime("2026-03-01T12:30:00Z")).toMatch(/in /);

    vi.useRealTimers();
  });
});

describe("shortPath", () => {
  it("leaves a short path alone", () => {
    expect(shortPath("/Users/notes")).toBe("/Users/notes");
  });

  it("keeps the last segments and elides the rest", () => {
    expect(shortPath("/Users/me/Documents/research/papers")).toBe("…/research/papers");
    expect(shortPath("/Users/me/Documents/research/papers", 1)).toBe("…/papers");
  });

  it("does not trip over a trailing slash", () => {
    expect(shortPath("/a/b/c/d/")).toBe("…/c/d");
  });
});

describe("duration", () => {
  it("reports seconds, then minutes, then hours", () => {
    expect(duration(45)).toBe("45s");
    expect(duration(125)).toBe("2m 05s");
    expect(duration(3600 * 3 + 240)).toBe("3h 04m");
  });

  it("shows a dash for a process that is not up", () => {
    expect(duration(0)).toBe("—");
    expect(duration(null)).toBe("—");
    expect(duration(-5)).toBe("—");
  });
});
