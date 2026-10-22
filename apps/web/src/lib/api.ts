import type {
  AnalyzeRequest,
  AnalyzeResponse,
  DetectRequest,
  DetectResponse,
  ModelsResponse,
} from "./types";

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/+$/, "");

const UNREACHABLE = "Can't reach the analysis server.";
// What a dev/preview proxy or gateway answers when the API itself is down.
const GATEWAY_STATUSES = new Set([502, 503, 504]);

/** A failed request. `status` is 0 when no response arrived at all. */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details: unknown;

  constructor(status: number, code: string, message: string, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }

  get isNetwork(): boolean {
    return this.code === "network_error";
  }
}

export function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      ...init,
      headers: { Accept: "application/json", ...(init.body ? { "Content-Type": "application/json" } : {}) },
    });
  } catch (error) {
    if (isAbort(error)) throw error;
    throw new ApiError(0, "network_error", UNREACHABLE);
  }

  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const error = (body as { error?: { code?: string; message?: string; details?: unknown } } | null)?.error;
    if (!error?.code && GATEWAY_STATUSES.has(response.status)) {
      throw new ApiError(response.status, "network_error", UNREACHABLE);
    }
    throw new ApiError(
      response.status,
      error?.code ?? "http_error",
      error?.message ?? `Request failed (${response.status}).`,
      error?.details,
    );
  }
  return body as T;
}

export const api = {
  models(signal?: AbortSignal): Promise<ModelsResponse> {
    return request("/api/v1/models", { signal });
  },
  detect(body: DetectRequest, signal?: AbortSignal): Promise<DetectResponse> {
    return request("/api/v1/detect", { method: "POST", body: JSON.stringify(body), signal });
  },
  analyze(body: AnalyzeRequest, signal?: AbortSignal): Promise<AnalyzeResponse> {
    return request("/api/v1/analyze", { method: "POST", body: JSON.stringify(body), signal });
  },
};
