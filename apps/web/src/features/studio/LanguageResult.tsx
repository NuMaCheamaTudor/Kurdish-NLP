import { Check, CircleDashed, Copy, GitCompareArrows, Info, RotateCcw, TriangleAlert } from "lucide-react";

import { Button, IconButton } from "../../components/Button";
import { Code } from "../../components/StatusBadge";
import { useCopy } from "../../hooks/useCopy";
import type { RequestState } from "../../hooks/useLatestRequest";
import { cx } from "../../lib/cx";
import { formatMs, formatPercent } from "../../lib/format";
import type { AnalyzeResponse, DetectResponse } from "../../lib/types";
import { ScoreList } from "./ScoreList";
import { detectionSummary } from "./summary";

export interface CompareProps {
  enabled: boolean;
  otherName: string | null;
  state: RequestState<DetectResponse>;
  onToggle: () => void;
}

interface LanguageResultProps {
  response: AnalyzeResponse;
  detection: DetectResponse;
  stale: boolean;
  onRerun: () => void;
  onChooseManually: () => void;
  compare: CompareProps;
}

export function LanguageResult({ response, detection, stale, onRerun, onChooseManually, compare }: LanguageResultProps) {
  const { copied, copy } = useCopy();
  const routing = response.language;
  const comparison = compare.enabled && compare.state.status === "success" ? compare.state.data : null;
  const top = detection.scores[0] ?? null;

  return (
    <div className="animate-[fade-in_160ms_var(--ease-out-soft)]">
      <div className="flex items-center justify-between gap-3">
        <p className="text-[12.5px] font-medium text-ink-3">
          {detection.abstained ? "Result" : "Detected language"}
        </p>
        <div className="-mr-1.5 flex items-center gap-1">
          {stale && (
            <Button variant="ghost" size="sm" onClick={onRerun} className="text-ink-3">
              <RotateCcw aria-hidden className="size-3.5" />
              Outdated
            </Button>
          )}
          <IconButton
            label={copied ? "Copied" : "Copy result"}
            tip={copied ? "Copied" : "Copy result"}
            tipAlign="end"
            onClick={() => void copy(detectionSummary(detection, routing))}
          >
            {copied ? (
              <Check aria-hidden className="size-4 text-accent" strokeWidth={2} />
            ) : (
              <Copy aria-hidden className="size-4" strokeWidth={1.75} />
            )}
          </IconButton>
        </div>
      </div>

      {detection.abstained ? (
        <div className="mt-3">
          <div className="flex items-start gap-3">
            <CircleDashed aria-hidden className="mt-1.5 size-5 shrink-0 text-ink-3" strokeWidth={1.75} />
            <div className="min-w-0">
              <h2 className="text-[24px] leading-tight font-semibold tracking-[-0.02em]">Inconclusive</h2>
              <p className="mt-1 text-[13.5px] leading-relaxed text-ink-2">{detection.reason_message}</p>
            </div>
          </div>
          <Button size="sm" className="mt-4" onClick={onChooseManually}>
            Choose language manually
          </Button>
        </div>
      ) : (
        <div className="mt-2 flex items-end justify-between gap-4">
          <div className="min-w-0">
            <h2 className="truncate text-[28px] leading-tight font-semibold tracking-[-0.025em]">
              {detection.language.name}
            </h2>
            <p className="mt-1 flex items-baseline gap-1.5 text-[13px] text-ink-3">
              {detection.language.alt_name && <span>{detection.language.alt_name}</span>}
              {detection.language.alt_name && <span aria-hidden>·</span>}
              <Code className="text-[12px]">{detection.language.code}</Code>
            </p>
          </div>
          <div className="shrink-0 text-right">
            <p className="tabular text-[28px] leading-tight font-semibold tracking-[-0.025em]">
              {formatPercent(detection.confidence)}
            </p>
            <p className="mt-1 text-[12px] text-ink-3">
              {detection.policy.calibrated ? "Calibrated estimate" : "Raw model score"}
            </p>
          </div>
        </div>
      )}

      {(routing.mode === "manual" || detection.warnings.length > 0) && (
        <ul className="mt-4 space-y-1.5">
          {routing.mode === "manual" && (
            <Note>
              Manual language: <span className="font-medium text-ink">{routing.name}</span>
              {routing.code !== detection.language.code && !detection.abstained
                ? ". It differs from the model estimate."
                : "."}
            </Note>
          )}
          {detection.warnings.map((warning) => (
            <Note key={warning.code} tone={warning.severity}>
              {warning.message}
            </Note>
          ))}
        </ul>
      )}

      {compare.enabled && (
        <CompareLine
          detection={detection}
          otherName={compare.otherName}
          state={compare.state}
        />
      )}

      {detection.scores.length > 0 && (
        <div className="mt-5 border-t border-line pt-4">
          <div className="mb-2 flex items-baseline justify-between">
            <h3 className="text-[12.5px] font-medium text-ink-3">
              {detection.abstained ? "Closest candidates" : "Scores"}
            </h3>
            <span className="text-[11.5px] text-ink-3">Model estimates, not guarantees</span>
          </div>
          <ScoreList
            scores={detection.scores}
            highlight={detection.abstained ? null : (top?.code ?? null)}
            primaryLabel={detection.model.name}
            compare={comparison ? { label: comparison.model.name, scores: comparison.scores } : null}
          />
        </div>
      )}

      <div className="mt-5 flex flex-wrap items-center justify-between gap-x-3 gap-y-2 border-t border-line pt-3">
        <p className="flex min-w-0 flex-wrap items-center gap-x-1.5 text-[12px] text-ink-3">
          <span data-testid="result-model" className="font-medium text-ink-2">
            {detection.model.name}
          </span>
          <span aria-hidden>·</span>
          <span>v{detection.model.version}</span>
          <span aria-hidden>·</span>
          <span title="Abstention policy">{detection.policy.name === "tuned" ? "Tuned" : "Reference"}</span>
          <span aria-hidden>·</span>
          <span className="tabular">{formatMs(detection.elapsed_ms)}</span>
        </p>
        {compare.otherName && (
          <Button
            variant="ghost"
            size="sm"
            aria-pressed={compare.enabled}
            onClick={compare.onToggle}
            className={cx("-mr-1.5", compare.enabled && "text-accent-strong")}
          >
            <GitCompareArrows aria-hidden className="size-3.5" />
            Compare
          </Button>
        )}
      </div>
    </div>
  );
}

function Note({ children, tone = "info" }: { children: React.ReactNode; tone?: "info" | "warning" }) {
  const Icon = tone === "warning" ? TriangleAlert : Info;
  return (
    <li
      className={cx(
        "flex gap-2 rounded-lg px-3 py-2 text-[12.5px] leading-relaxed",
        tone === "warning" ? "bg-warn-soft text-warn" : "bg-wash text-ink-2",
      )}
    >
      <Icon aria-hidden className="mt-[3px] size-3.5 shrink-0 opacity-80" strokeWidth={1.75} />
      <span>{children}</span>
    </li>
  );
}

function CompareLine({
  detection,
  otherName,
  state,
}: {
  detection: DetectResponse;
  otherName: string | null;
  state: RequestState<DetectResponse>;
}) {
  const other = state.data;
  let content: React.ReactNode;
  if (state.status === "error") {
    content = <span>{state.error?.message ?? "Comparison failed."}</span>;
  } else if (!other || state.status === "loading") {
    content = <span>Running {otherName}…</span>;
  } else {
    const agree = other.language.code === detection.language.code;
    content = (
      <>
        <span className="font-medium text-ink-2">{other.model.name}</span>
        <span aria-hidden>→</span>
        <span className="text-ink">{other.abstained ? "Inconclusive" : other.language.name}</span>
        {!other.abstained && <span className="tabular">{formatPercent(other.confidence)}</span>}
        <span
          className={cx(
            "ml-auto inline-flex items-center gap-1 font-medium",
            agree ? "text-accent-strong" : "text-warn",
          )}
        >
          {agree ? <Check aria-hidden className="size-3.5" /> : <TriangleAlert aria-hidden className="size-3.5" />}
          {agree ? "Agrees" : "Disagrees"}
        </span>
      </>
    );
  }
  return (
    <div
      aria-live="polite"
      className="mt-4 flex flex-wrap items-center gap-x-1.5 gap-y-1 rounded-lg border border-line px-3 py-2 text-[12.5px] text-ink-3"
    >
      {content}
    </div>
  );
}
