import { ArrowRight, CircleDashed, LoaderCircle, RotateCcw, TriangleAlert, WifiOff } from "lucide-react";

import { Button } from "../../components/Button";
import { Tag } from "../../components/StatusBadge";
import type { ApiError } from "../../lib/api";
import { Link } from "../../lib/router";
import type { AnalyzeResponse, LanguageDetail, Task, TaskInfo } from "../../lib/types";
import { EntityResult } from "./EntityResult";
import { GrammarResult } from "./GrammarResult";
import { LanguageResult, type CompareProps } from "./LanguageResult";

const TASK_TITLES: Record<Task, string> = {
  language: "Language",
  entities: "Named entities",
  grammar: "Grammar",
};

const EMPTY_HINTS: Record<Task, string> = {
  language: "The detected language and per-language scores appear here.",
  entities: "Entities appear here, highlighted in your text.",
  grammar: "Tokens, lemmas, tags and dependencies appear here.",
};

export type PanelView =
  | { kind: "empty" }
  | { kind: "loading" }
  | { kind: "unavailable"; task: Task; message: string }
  | { kind: "error"; error: ApiError }
  | { kind: "result"; response: AnalyzeResponse; text: string; stale: boolean };

interface ResultPanelProps {
  view: PanelView;
  task: Task;
  busy: boolean;
  showSpinner: boolean;
  languages: LanguageDetail[];
  tasks: TaskInfo[] | undefined;
  onRetry: () => void;
  onRerun: () => void;
  onUseLanguageTask: () => void;
  onChooseManually: () => void;
  compare: CompareProps;
}

export function ResultPanel(props: ResultPanelProps) {
  const { view } = props;
  return (
    <section
      aria-label="Results"
      aria-busy={props.busy}
      className="rounded-xl border border-line bg-surface p-5 shadow-card"
    >
      <div className={props.busy && view.kind === "result" ? "opacity-60 transition-opacity duration-150" : undefined}>
        <PanelBody {...props} />
      </div>
    </section>
  );
}

function PanelBody({ view, task, showSpinner, languages, onRetry, onRerun, onUseLanguageTask, onChooseManually, compare }: ResultPanelProps) {
  switch (view.kind) {
    case "unavailable":
      return (
        <div className="animate-[fade-in_160ms_var(--ease-out-soft)]">
          <p className="flex items-center gap-1.5 text-[12.5px] font-medium text-ink-3">
            <CircleDashed aria-hidden className="size-3.5" strokeWidth={2} />
            Unavailable
          </p>
          <h2 className="mt-3 text-[17px] font-semibold tracking-[-0.01em]">{TASK_TITLES[view.task]}</h2>
          <p className="mt-1 text-[13.5px] text-ink-2">{view.message}</p>
          <div className="mt-4 flex flex-wrap items-center gap-2">
            <Button size="sm" onClick={onUseLanguageTask}>
              Detect language instead
            </Button>
            <Link
              to="models"
              className="inline-flex h-7 items-center gap-1 rounded-lg px-2 text-[12.5px] font-medium text-ink-2 transition-colors duration-150 hover:bg-wash hover:text-ink"
            >
              Models
              <ArrowRight aria-hidden className="size-3.5" />
            </Link>
          </div>
        </div>
      );

    case "error": {
      const network = view.error.isNetwork;
      const Icon = network ? WifiOff : TriangleAlert;
      return (
        <div role="alert" className="animate-[fade-in_160ms_var(--ease-out-soft)]">
          <p className="flex items-center gap-1.5 text-[12.5px] font-medium text-danger">
            <Icon aria-hidden className="size-3.5" strokeWidth={2} />
            {network ? "Server unavailable" : "Couldn't analyze"}
          </p>
          <p className="mt-3 text-[14px] text-ink">{view.error.message}</p>
          {network && <p className="mt-1 text-[13px] text-ink-3">Check that the API is running.</p>}
          {(network || view.error.status >= 500) && (
            <Button size="sm" className="mt-4" onClick={onRetry}>
              <RotateCcw aria-hidden className="size-3.5" />
              Retry
            </Button>
          )}
        </div>
      );
    }

    case "result": {
      const { response } = view;
      if (response.entities) return <EntityResult text={view.text} result={response.entities} />;
      if (response.grammar) return <GrammarResult result={response.grammar} />;
      if (response.detection) {
        return (
          <LanguageResult
            response={response}
            detection={response.detection}
            stale={view.stale}
            onRerun={onRerun}
            onChooseManually={onChooseManually}
            compare={compare}
          />
        );
      }
      return null;
    }

    case "loading":
    case "empty":
      return (
        <div>
          <p className="flex items-center gap-2 text-[13.5px] font-medium text-ink">
            {TASK_TITLES[task]}
            {view.kind === "loading" && showSpinner && (
              <LoaderCircle aria-label="Analyzing" className="size-3.5 animate-spin text-ink-3" />
            )}
          </p>
          <p className="mt-1 text-[13px] leading-relaxed text-ink-3">{EMPTY_HINTS[task]}</p>
          {languages.length > 0 && (
            <div className="mt-5 border-t border-line pt-4">
              <p className="mb-2 text-[12px] text-ink-3">Supported languages</p>
              <ul className="flex flex-wrap gap-1.5">
                {languages.map((language) => (
                  <li key={language.code}>
                    <Tag>{language.name}</Tag>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      );
  }
}
