import { useCallback, useEffect } from "react";

import { api } from "../lib/api";
import type { ModelsResponse } from "../lib/types";
import { useLatestRequest } from "./useLatestRequest";

/** What the server can do right now: models, tasks, languages and limits. */
export function useCapabilities() {
  const request = useLatestRequest<ModelsResponse>();
  const { run } = request;
  const load = useCallback(() => void run((signal) => api.models(signal)), [run]);

  useEffect(load, [load]);

  return { ...request, retry: load };
}

export type Capabilities = ReturnType<typeof useCapabilities>;
