import { useCallback, useState } from "react";

// Per-viewer conveniences only. Storage can be unavailable (private windows,
// blocked site data), so every access is guarded and the app works without it.

type StorageKind = "local" | "session";

function storage(kind: StorageKind): Storage | null {
  try {
    return kind === "local" ? window.localStorage : window.sessionStorage;
  } catch {
    return null;
  }
}

function read<T>(kind: StorageKind, key: string, fallback: T, valid: (value: unknown) => boolean): T {
  try {
    const raw = storage(kind)?.getItem(key);
    if (raw == null) return fallback;
    const value: unknown = JSON.parse(raw);
    return valid(value) ? (value as T) : fallback;
  } catch {
    return fallback;
  }
}

export function useStoredState<T>(
  key: string,
  fallback: T,
  options: { kind?: StorageKind; valid?: (value: unknown) => boolean } = {},
) {
  const kind = options.kind ?? "local";
  const valid = options.valid ?? ((value: unknown) => typeof value === typeof fallback);
  const [value, setValue] = useState<T>(() => read(kind, key, fallback, valid));

  const update = useCallback(
    (next: T) => {
      setValue(next);
      try {
        storage(kind)?.setItem(key, JSON.stringify(next));
      } catch {
        // Quota or access errors only cost persistence.
      }
    },
    [key, kind],
  );

  return [value, update] as const;
}
