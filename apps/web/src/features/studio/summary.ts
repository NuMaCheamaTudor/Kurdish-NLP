import { formatPercent } from "../../lib/format";
import type { AnalyzeResponse, DetectResponse } from "../../lib/types";

/** Plain-text summary of a detection, for the clipboard. */
export function detectionSummary(detection: DetectResponse, routing?: AnalyzeResponse["language"]): string {
  const { language, model, policy } = detection;
  const lines: string[] = [];
  if (detection.abstained) {
    lines.push(`Language: Inconclusive (und, ${detection.reason})`);
    if (detection.reason_message) lines.push(detection.reason_message);
  } else {
    const alt = language.alt_name ? ` (${language.alt_name})` : "";
    lines.push(`Language: ${language.name}${alt} [${language.code}]`);
    lines.push(`Score: ${formatPercent(detection.confidence, 2)} (${policy.calibrated ? "calibrated" : "raw"} model estimate)`);
  }
  if (routing?.mode === "manual") lines.push(`Manual language: ${routing.name} [${routing.code}]`);
  lines.push(`Model: ${model.name} (${model.identifier} ${model.version}), ${policy.name} policy`);
  if (detection.scores.length) {
    lines.push(
      `Scores: ${detection.scores.map((item) => `${item.code} ${formatPercent(item.score, 2)}`).join(", ")}`,
    );
  }
  for (const warning of detection.warnings) lines.push(`Note: ${warning.message}`);
  return lines.join("\n");
}
