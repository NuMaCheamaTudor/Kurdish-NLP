import { vi } from "vitest";

import analyzeCkbFasttext from "./fixtures/analyze-ckb-fasttext.json";
import analyzeCkbTfidf from "./fixtures/analyze-ckb-tfidf.json";
import analyzeInconclusive from "./fixtures/analyze-inconclusive.json";
import detectCkbFasttext from "./fixtures/detect-ckb-fasttext.json";
import taskUnavailable from "./fixtures/error-task-unavailable.json";
import models from "./fixtures/models.json";

// Fixtures are responses recorded with curl from the real API serving the
// baseline-v1 models, using Dataset V1 train sentences or short typed text.
export const fixtures = {
  models,
  analyzeCkbTfidf,
  analyzeCkbFasttext,
  analyzeInconclusive,
  detectCkbFasttext,
  taskUnavailable,
};

export interface RecordedCall {
  method: string;
  path: string;
  body: Record<string, unknown> | null;
}

type Reply = { status?: number; body: unknown } | Promise<{ status?: number; body: unknown }> | Error;
type Handler = (call: RecordedCall) => Reply;

export function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

/** Route fetch calls to handlers keyed by "METHOD /path"; records every call. */
export function mockApi(routes: Record<string, Handler>) {
  const calls: RecordedCall[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), "http://localhost");
    const method = (init?.method ?? "GET").toUpperCase();
    const call: RecordedCall = {
      method,
      path: url.pathname,
      body: typeof init?.body === "string" ? (JSON.parse(init.body) as Record<string, unknown>) : null,
    };
    calls.push(call);
    const handler = routes[`${method} ${url.pathname}`];
    if (!handler) return json({ error: { code: "not_found", message: "Not found." } }, 404);
    const reply = handler(call);
    if (reply instanceof Error) throw reply;
    const signal = init?.signal;
    const settled = await Promise.race([
      Promise.resolve(reply),
      new Promise<never>((_, reject) => {
        if (signal?.aborted) reject(new DOMException("Aborted", "AbortError"));
        signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
      }),
    ]);
    return json(settled.body, settled.status ?? 200);
  });
  vi.stubGlobal("fetch", fetchMock);
  return { calls, fetchMock };
}

export function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}
