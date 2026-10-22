import { useCallback, useEffect, useRef, useState } from "react";

import { Composer, LANGUAGE_SELECT_ID } from "../features/studio/Composer";
import type { Direction } from "../features/studio/OptionMenus";
import { ResultPanel, type PanelView } from "../features/studio/ResultPanel";
import type { Capabilities } from "../hooks/useCapabilities";
import { useDelayedFlag } from "../hooks/useDelayedFlag";
import { useLatestRequest } from "../hooks/useLatestRequest";
import { useStoredState } from "../hooks/useStoredState";
import { api } from "../lib/api";
import { formatNumber, formatPercent } from "../lib/format";
import { codePointLength } from "../lib/text";
import type { AnalyzeResponse, DetectResponse, PolicyName, Task } from "../lib/types";

interface RunParams {
  text: string;
  task: Task;
  language: string;
  model: string | null;
  policy: PolicyName;
}

const isTask = (value: unknown) => value === "language" || value === "entities" || value === "grammar";
const isPolicy = (value: unknown) => value === "tuned" || value === "reference";
const isDirection = (value: unknown) => value === "auto" || value === "rtl" || value === "ltr";

export function StudioPage({ capabilities }: { capabilities: Capabilities }) {
  const caps = capabilities.data;
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // The draft survives a reload of this tab only; settings persist per browser.
  const [text, setText] = useStoredState("knlp.text", "", { kind: "session" });
  const [task, setTask] = useStoredState<Task>("knlp.task", "language", { valid: isTask });
  const [storedLanguage, setLanguage] = useStoredState("knlp.language", "auto");
  const [modelPreference, setModelPreference] = useStoredState("knlp.model", "");
  const [policy, setPolicy] = useStoredState<PolicyName>("knlp.policy", "tuned", { valid: isPolicy });
  const [direction, setDirection] = useStoredState<Direction>("knlp.direction", "auto", { valid: isDirection });
  const [compareOn, setCompareOn] = useStoredState("knlp.compare", false);

  const analysis = useLatestRequest<AnalyzeResponse>();
  const comparison = useLatestRequest<DetectResponse>();
  const [lastRun, setLastRun] = useState<RunParams | null>(null);

  const languages = caps?.languages ?? [];
  const languageModels = caps?.models.filter((model) => model.task === "language") ?? [];
  const readyModels = languageModels.filter((model) => model.status === "ready");
  const model = readyModels.some((item) => item.id === modelPreference)
    ? modelPreference
    : (caps?.default_model ?? null);
  const language =
    !caps || storedLanguage === "auto" || languages.some((item) => item.code === storedLanguage)
      ? storedLanguage
      : "auto";
  const taskInfo = caps?.tasks.find((item) => item.id === task);
  const maxChars = caps?.limits.max_text_chars ?? 5000;

  const blockReason = useCallback(
    (params: RunParams): string | null => {
      if (!params.text.trim()) return "Enter text to analyze";
      if (codePointLength(params.text) > maxChars) return `Text is longer than ${formatNumber(maxChars)} characters`;
      if (capabilities.status !== "success" || !caps) return "Server unavailable";
      const info = caps.tasks.find((item) => item.id === params.task);
      if (info && !info.available) return info.message ?? "Unavailable";
      if ((params.task === "language" || params.language === "auto") && !params.model) {
        return "No language ID model is available";
      }
      return null;
    },
    [capabilities.status, caps, maxChars],
  );

  const current: RunParams = { text, task, language, model, policy };
  const block = blockReason(current);

  const compareRef = useRef(compareOn);
  compareRef.current = compareOn;

  const runCompare = useCallback(
    (params: RunParams, detection: DetectResponse) => {
      const other = readyModels.find((item) => item.id !== detection.model.id);
      if (!other) return;
      void comparison.run((signal) =>
        api.detect({ text: params.text, model: other.id, policy: params.policy, top_k: 7 }, signal),
      );
    },
    // readyModels is derived from caps, so caps is the real dependency.
    [caps, comparison.run],
  );

  const analyze = useCallback(
    (override?: Partial<RunParams>) => {
      const params = { ...current, ...override };
      if (blockReason(params)) return;
      setLastRun(params);
      comparison.reset();
      void analysis
        .run((signal) =>
          api.analyze(
            {
              text: params.text,
              task: params.task,
              language: params.language,
              model: params.model ?? undefined,
              policy: params.policy,
            },
            signal,
          ),
        )
        .then((response) => {
          if (response?.detection && params.task === "language" && compareRef.current) {
            runCompare(params, response.detection);
          }
        });
    },
    [text, task, language, model, policy, blockReason, analysis.run, comparison.reset, runCompare],
  );

  // Switching model or policy re-runs the last analysis on the same text.
  const previousSettings = useRef({ model, policy });
  useEffect(() => {
    const previous = previousSettings.current;
    previousSettings.current = { model, policy };
    if (previous.model === model && previous.policy === policy) return;
    if (lastRun && lastRun.text === text && analysis.status !== "idle") analyze();
  }, [model, policy]);

  // ⌘/Ctrl + Enter anywhere on the page.
  const analyzeRef = useRef(analyze);
  analyzeRef.current = analyze;
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Enter" && (event.metaKey || event.ctrlKey) && !event.isComposing) {
        event.preventDefault();
        analyzeRef.current();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const clear = () => {
    setText("");
    setLastRun(null);
    analysis.reset();
    comparison.reset();
    textareaRef.current?.focus();
  };

  const toggleCompare = () => {
    const next = !compareOn;
    setCompareOn(next);
    const detection = analysis.data?.detection;
    if (next && lastRun && detection) runCompare(lastRun, detection);
    else comparison.reset();
  };

  const chooseManually = () => {
    const select = document.getElementById(LANGUAGE_SELECT_ID) as HTMLSelectElement | null;
    select?.focus();
    try {
      select?.showPicker?.();
    } catch {
      // showPicker needs a user gesture in some browsers; focus is enough.
    }
  };

  const busy = analysis.status === "loading";
  const showSpinner = useDelayedFlag(busy);
  const stale =
    lastRun !== null && (lastRun.text !== text || lastRun.task !== task || lastRun.language !== language);

  let view: PanelView;
  if (taskInfo && !taskInfo.available) {
    view = { kind: "unavailable", task, message: taskInfo.message ?? "Unavailable." };
  } else if (analysis.status === "error" && analysis.error) {
    view =
      analysis.error.code === "task_unavailable" && lastRun
        ? { kind: "unavailable", task: lastRun.task, message: analysis.error.message }
        : { kind: "error", error: analysis.error };
  } else if (analysis.data && lastRun) {
    view = { kind: "result", response: analysis.data, text: lastRun.text, stale };
  } else if (capabilities.status === "error" && capabilities.error) {
    view = { kind: "error", error: capabilities.error };
  } else {
    view = { kind: busy ? "loading" : "empty" };
  }

  const detection = analysis.data?.detection;
  const announcement =
    analysis.status === "success" && detection
      ? detection.abstained
        ? "Result: inconclusive."
        : `Detected ${detection.language.name}, ${formatPercent(detection.confidence)}.`
      : "";

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-[22px] leading-tight font-semibold tracking-[-0.02em]">Analyze text</h1>
        <p className="mt-1 text-[14px] text-ink-3">Language detection and linguistic analysis.</p>
      </div>

      <div className="grid grid-cols-1 items-start gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(340px,400px)]">
        <Composer
          textareaRef={textareaRef}
          text={text}
          onTextChange={setText}
          onClear={clear}
          onExample={(example) => {
            setText(example);
            analyze({ text: example });
          }}
          task={task}
          tasks={caps?.tasks}
          onTaskChange={setTask}
          language={language}
          languages={languages}
          onLanguageChange={setLanguage}
          model={model}
          models={languageModels}
          onModelChange={setModelPreference}
          policy={policy}
          onPolicyChange={setPolicy}
          direction={direction}
          onDirectionChange={setDirection}
          maxChars={maxChars}
          canAnalyze={block === null}
          busy={showSpinner}
          analyzeHint={block ?? undefined}
          onAnalyze={() => analyze()}
        />
        <div className="lg:sticky lg:top-20">
          <ResultPanel
            view={view}
            task={task}
            busy={busy}
            showSpinner={showSpinner}
            languages={languages}
            tasks={caps?.tasks}
            onRetry={() => (capabilities.status === "error" ? capabilities.retry() : analyze(lastRun ?? undefined))}
            onRerun={() => analyze()}
            onUseLanguageTask={() => setTask("language")}
            onChooseManually={chooseManually}
            compare={{
              enabled: compareOn,
              otherName:
                readyModels.length > 1
                  ? (readyModels.find((item) => item.id !== detection?.model.id)?.name ?? null)
                  : null,
              state: comparison,
              onToggle: toggleCompare,
            }}
          />
        </div>
      </div>
      <p className="sr-only" aria-live="polite">
        {announcement}
      </p>
    </div>
  );
}
