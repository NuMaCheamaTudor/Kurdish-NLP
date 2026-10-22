import { describe, expect, it } from "vitest";

import { formatPercent } from "./format";
import {
  codePointLength,
  firstStrongDirection,
  hasMixedDirections,
  segmentByCodePoints,
  sliceByCodePoints,
} from "./text";

describe("code point helpers", () => {
  it("counts code points like Python len()", () => {
    expect(codePointLength("کوردی")).toBe(5);
    expect(codePointLength("a👋b")).toBe(3); // "a👋b".length is 4 in UTF-16
    expect(codePointLength("می‌خواهم")).toBe(8); // ZWNJ counts and is kept
  });

  it("slices and segments by code point offsets", () => {
    const text = "سڵاو 👋 لە هەولێر";
    const start = Array.from(text).indexOf("ه");
    expect(sliceByCodePoints(text, start, start + 6)).toBe("هەولێر");
    const segments = segmentByCodePoints(text, [{ start, end: start + 6, value: "LOC" }]);
    expect(segments.map((segment) => segment.text).join("")).toBe(text);
    expect(segments.at(-1)).toEqual({ text: "هەولێر", span: { start, end: start + 6, value: "LOC" } });
  });

  it("skips overlapping or out-of-range spans instead of guessing", () => {
    const segments = segmentByCodePoints("abcdef", [
      { start: 0, end: 3, value: 1 },
      { start: 2, end: 4, value: 2 },
      { start: 5, end: 99, value: 3 },
    ]);
    expect(segments.filter((segment) => segment.span).map((segment) => segment.span!.value)).toEqual([1]);
    expect(segments.map((segment) => segment.text).join("")).toBe("abcdef");
  });

  it("finds the first strong direction", () => {
    expect(firstStrongDirection("  123 ئەمڕۆ Kurdish")).toBe("rtl");
    expect(firstStrongDirection("(Ez) دەڕۆم")).toBe("ltr");
    expect(firstStrongDirection("123 !!")).toBeNull();
    expect(hasMixedDirections("ئەمڕۆ Kurdish")).toBe(true);
  });
});

describe("formatPercent", () => {
  it("never rounds a non-zero score to 0% or a non-certain score to 100%", () => {
    expect(formatPercent(0)).toBe("0%");
    expect(formatPercent(1e-9)).toBe("<0.1%");
    expect(formatPercent(0.9999999)).toBe(">99.9%");
    expect(formatPercent(1)).toBe("100.0%");
    expect(formatPercent(0.86712)).toBe("86.7%");
  });
});
