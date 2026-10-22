import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, isAbort } from "../lib/api";

export type RequestStatus = "idle" | "loading" | "success" | "error";

export interface RequestState<T> {
  status: RequestStatus;
  data: T | null;
  error: ApiError | null;
}

const IDLE = { status: "idle", data: null, error: null } as const;

function toApiError(error: unknown): ApiError {
  return error instanceof ApiError ? error : new ApiError(0, "unexpected", "Something went wrong.");
}

/**
 * Runs one request at a time. Starting a new request aborts the previous one,
 * and a response that arrives after a newer request started is discarded, so
 * the UI never shows a stale result.
 */
export function useLatestRequest<T>() {
  const [state, setState] = useState<RequestState<T>>(IDLE);
  const controller = useRef<AbortController | null>(null);
  const sequence = useRef(0);

  const run = useCallback(async (fetcher: (signal: AbortSignal) => Promise<T>): Promise<T | null> => {
    controller.current?.abort();
    const current = new AbortController();
    controller.current = current;
    const id = ++sequence.current;
    // Keep the previous data visible while loading; it is dimmed, not removed.
    setState((previous) => ({ status: "loading", data: previous.data, error: null }));
    try {
      const data = await fetcher(current.signal);
      if (id !== sequence.current) return null;
      setState({ status: "success", data, error: null });
      return data;
    } catch (error) {
      if (isAbort(error) || id !== sequence.current) return null;
      setState({ status: "error", data: null, error: toApiError(error) });
      return null;
    }
  }, []);

  const reset = useCallback(() => {
    controller.current?.abort();
    sequence.current += 1;
    setState(IDLE);
  }, []);

  useEffect(() => () => controller.current?.abort(), []);

  return { ...state, run, reset };
}
