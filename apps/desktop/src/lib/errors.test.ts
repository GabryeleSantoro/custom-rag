import { describe, expect, it } from "vitest";
import { errorText } from "./errors";
import { IpcError, parseIpcError } from "./ipc";

const t = (key: string, opts?: Record<string, unknown>) => `T(${key}:${JSON.stringify(opts)})`;
const known = (key: string) => key === "errors.model_not_installed";

describe("parseIpcError", () => {
  it("reads a coded JSON payload", () => {
    const e = parseIpcError('{"code":"model_not_installed","message":"m","params":{"a":1}}');
    expect(e.code).toBe("model_not_installed");
    expect(e.params).toEqual({ a: 1 });
    expect(e.message).toBe("m");
  });
  it("keeps a plain string as the message", () => {
    const e = parseIpcError("404: something");
    expect(e.code).toBeUndefined();
    expect(e.message).toBe("404: something");
  });
  it("treats JSON without a string code as plain text", () => {
    const e = parseIpcError('{"detail":"x"}');
    expect(e.code).toBeUndefined();
    expect(e.message).toBe('{"detail":"x"}');
  });
});

describe("errorText", () => {
  it("translates a known code with its params", () => {
    const err = new IpcError("raw", "model_not_installed", { a: 1 });
    expect(errorText(err, t, known)).toBe('T(errors.model_not_installed:{"a":1})');
  });
  it("falls back to the English message for a code the UI has no key for", () => {
    expect(errorText(new IpcError("raw english", "brand_new_code"), t, known)).toBe("raw english");
  });
  it("handles errors without a code and non-errors", () => {
    expect(errorText(new Error("x"), t, known)).toBe("x");
    expect(errorText("y", t, known)).toBe("y");
  });
});
