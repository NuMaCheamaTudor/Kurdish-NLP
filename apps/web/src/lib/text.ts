// Text helpers. Nothing here changes the user's text; they only measure it.

/** Length in Unicode code points, matching Python's len() on the server. */
export function codePointLength(text: string): number {
  let count = 0;
  for (const _ of text) count += 1;
  return count;
}

/** Split text into segments by code point offsets (as returned by the API). */
export function sliceByCodePoints(text: string, start: number, end: number): string {
  return Array.from(text).slice(start, end).join("");
}

export interface Span<T> {
  start: number;
  end: number;
  value: T;
}

export type Segment<T> = { text: string; span: Span<T> | null };

/**
 * Split `text` into plain and highlighted segments. Spans use code point
 * offsets; overlapping or out-of-range spans are skipped rather than guessed.
 */
export function segmentByCodePoints<T>(text: string, spans: Span<T>[]): Segment<T>[] {
  const chars = Array.from(text);
  const sorted = [...spans].sort((a, b) => a.start - b.start);
  const segments: Segment<T>[] = [];
  let cursor = 0;
  for (const span of sorted) {
    if (span.start < cursor || span.end <= span.start || span.end > chars.length) continue;
    if (span.start > cursor) segments.push({ text: chars.slice(cursor, span.start).join(""), span: null });
    segments.push({ text: chars.slice(span.start, span.end).join(""), span });
    cursor = span.end;
  }
  if (cursor < chars.length) segments.push({ text: chars.slice(cursor).join(""), span: null });
  return segments;
}

const RTL_RANGES = /[֐-ࣿיִ-﷿ﹰ-﻿]/u;
const STRONG_LTR = /[A-Za-zÀ-ɏͰ-ϿЀ-ӿ]/u;

/** Direction of the first strong character, mirroring dir="auto". */
export function firstStrongDirection(text: string): "rtl" | "ltr" | null {
  for (const char of text) {
    if (RTL_RANGES.test(char)) return "rtl";
    if (STRONG_LTR.test(char)) return "ltr";
  }
  return null;
}

export function hasMixedDirections(text: string): boolean {
  return RTL_RANGES.test(text) && STRONG_LTR.test(text);
}
