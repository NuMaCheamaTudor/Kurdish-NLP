/** Percentage for a score in [0, 1]. Tiny non-zero values read as "<0.1%". */
export function formatPercent(score: number, digits = 1): string {
  if (score <= 0) return "0%";
  const percent = score * 100;
  const floor = 10 ** -digits;
  if (percent < floor) return `<${floor}%`;
  if (percent >= 100 - floor && score < 1) return `>${(100 - floor).toFixed(digits)}%`;
  return `${percent.toFixed(digits)}%`;
}

export function formatMetric(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(0)} kB`;
  return `${bytes} B`;
}

export function formatNumber(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

export function formatMs(ms: number): string {
  return ms < 10 ? `${ms.toFixed(1)} ms` : `${Math.round(ms)} ms`;
}

export const isMac =
  typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
