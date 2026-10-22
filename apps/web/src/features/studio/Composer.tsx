import { ClipboardPaste, CornerDownLeft, Languages, LoaderCircle, X } from "lucide-react";
import { useMemo, useState, type RefObject } from "react";

import { Button, IconButton } from "../../components/Button";
import { SelectField, type SelectOption } from "../../components/SelectField";
import { cx } from "../../lib/cx";
import { formatNumber, isMac } from "../../lib/format";
import { codePointLength, firstStrongDirection } from "../../lib/text";
import type { LanguageDetail, ModelInfo, PolicyName, Task, TaskInfo } from "../../lib/types";
import { EXAMPLE_SOURCE, EXAMPLES } from "./examples";
import { AdvancedMenu, DirectionMenu, type Direction } from "./OptionMenus";
import { TaskTabs } from "./TaskTabs";

export const LANGUAGE_SELECT_ID = "language-setting";

interface ComposerProps {
  textareaRef: RefObject<HTMLTextAreaElement | null>;
  text: string;
  onTextChange: (text: string) => void;
  onClear: () => void;
  onExample: (text: string) => void;
  task: Task;
  tasks: TaskInfo[] | undefined;
  onTaskChange: (task: Task) => void;
  language: string;
  languages: LanguageDetail[];
  onLanguageChange: (language: string) => void;
  model: string | null;
  models: ModelInfo[];
  onModelChange: (model: string) => void;
  policy: PolicyName;
  onPolicyChange: (policy: PolicyName) => void;
  direction: Direction;
  onDirectionChange: (direction: Direction) => void;
  maxChars: number;
  canAnalyze: boolean;
  busy: boolean;
  analyzeHint?: string;
  onAnalyze: () => void;
}

const canReadClipboard = typeof navigator !== "undefined" && typeof navigator.clipboard?.readText === "function";

export function Composer(props: ComposerProps) {
  const { textareaRef, text, onTextChange, maxChars } = props;
  const [pasteFailed, setPasteFailed] = useState(false);
  const length = codePointLength(text);
  const tooLong = length > maxChars;
  const effectiveDirection =
    props.direction === "auto" ? (firstStrongDirection(text) ?? "ltr") : props.direction;

  const languageOptions = useMemo<SelectOption[]>(
    () => [
      { value: "auto", label: "Auto-detect" },
      ...props.languages.map((language) => ({ value: language.code, label: language.name })),
    ],
    [props.languages],
  );
  const modelOptions = useMemo<SelectOption[]>(
    () =>
      props.models.map((model) => ({
        value: model.id,
        label: model.status === "ready" ? model.name : `${model.name} (unavailable)`,
        disabled: model.status !== "ready",
      })),
    [props.models],
  );
  const selectedModel = props.models.find((model) => model.id === props.model);

  const paste = async () => {
    const textarea = textareaRef.current;
    try {
      const clip = await navigator.clipboard.readText();
      const start = textarea?.selectionStart ?? text.length;
      const end = textarea?.selectionEnd ?? text.length;
      onTextChange(text.slice(0, start) + clip + text.slice(end));
      requestAnimationFrame(() => {
        textarea?.focus();
        textarea?.setSelectionRange(start + clip.length, start + clip.length);
      });
    } catch {
      setPasteFailed(true);
      textarea?.focus();
      window.setTimeout(() => setPasteFailed(false), 2400);
    }
  };

  return (
    <section
      aria-label="Input"
      className="rounded-xl border border-line bg-surface shadow-card transition-[border-color,box-shadow] duration-150 focus-within:border-line-strong"
    >
      <div className="flex items-center justify-between gap-1 border-b border-line pr-1.5 pl-1.5 sm:pr-2 sm:pl-2.5">
        <TaskTabs value={props.task} tasks={props.tasks} onChange={props.onTaskChange} />
        <div className="flex shrink-0 items-center">
          <DirectionMenu value={props.direction} effective={effectiveDirection} onChange={props.onDirectionChange} />
          {canReadClipboard && (
            <IconButton
              className="max-sm:hidden"
              label="Paste"
              tip={pasteFailed ? `Press ${isMac ? "⌘" : "Ctrl+"}V to paste` : "Paste"}
              onClick={paste}
            >
              <ClipboardPaste aria-hidden className="size-4" strokeWidth={1.75} />
            </IconButton>
          )}
          <IconButton label="Clear text" tip="Clear" tipAlign="end" onClick={props.onClear} disabled={!text}>
            <X aria-hidden className="size-4" strokeWidth={1.75} />
          </IconButton>
        </div>
      </div>

      <div className="relative">
        <textarea
          ref={textareaRef}
          id="studio-input"
          value={text}
          onChange={(event) => onTextChange(event.target.value)}
          dir={props.direction}
          aria-label="Text to analyze"
          aria-invalid={tooLong || undefined}
          aria-describedby="studio-count"
          placeholder="Type or paste text…"
          spellCheck={false}
          autoCorrect="off"
          autoCapitalize="off"
          autoComplete="off"
          className={cx(
            "user-text block max-h-[60vh] min-h-[200px] w-full resize-none bg-transparent px-5 pt-4 pb-3 text-ink outline-none placeholder:text-ink-3/70 sm:min-h-[260px]",
            "[field-sizing:content] focus-visible:outline-none",
          )}
        />
        {!text && (
          <div className="flex flex-wrap items-center gap-1.5 px-5 pb-4 animate-[fade-in_180ms_var(--ease-out-soft)]">
            <span className="mr-1 text-[12.5px] text-ink-3">Try</span>
            {EXAMPLES.map((example) => (
              <button
                key={example.id}
                type="button"
                onClick={() => props.onExample(example.text)}
                title={`Example from Dataset V1 training data (${EXAMPLE_SOURCE})`}
                className="h-7 rounded-full border border-line px-3 text-[12.5px] text-ink-2 transition-colors duration-150 hover:border-line-strong hover:bg-wash hover:text-ink"
              >
                {example.label}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-line px-3 py-2.5">
        <SelectField
          id={LANGUAGE_SELECT_ID}
          label="Language"
          icon={<Languages className="size-3.5" strokeWidth={1.75} />}
          value={props.language}
          options={languageOptions}
          onChange={props.onLanguageChange}
          disabled={!props.languages.length}
        />
        <SelectField
          label="Model"
          value={props.model ?? ""}
          options={modelOptions.length ? modelOptions : [{ value: "", label: "—" }]}
          onChange={props.onModelChange}
          disabled={!modelOptions.some((option) => !option.disabled)}
        />
        <AdvancedMenu
          policy={props.policy}
          policies={selectedModel?.policies ?? []}
          onPolicyChange={props.onPolicyChange}
        />

        <div className="ml-auto flex items-center gap-3">
          <span
            id="studio-count"
            className={cx("tabular text-[12px]", tooLong ? "font-medium text-danger" : "text-ink-3")}
          >
            {tooLong && <span className="sr-only">Too long: </span>}
            {formatNumber(length)} / {formatNumber(maxChars)}
          </span>
          <Button
            variant="primary"
            onClick={props.onAnalyze}
            disabled={!props.canAnalyze}
            title={props.analyzeHint}
            aria-keyshortcuts={isMac ? "Meta+Enter" : "Control+Enter"}
            className="min-w-[104px] pr-2"
          >
            {props.busy ? (
              <LoaderCircle aria-hidden className="size-3.5 animate-spin" />
            ) : null}
            Analyze
            <kbd className="ml-0.5 inline-flex items-center gap-px rounded bg-white/15 px-1 py-px font-sans text-[11px] font-medium text-white/85 max-sm:hidden">
              {isMac ? "⌘" : "Ctrl"}
              <CornerDownLeft aria-hidden className="size-3" strokeWidth={2} />
            </kbd>
          </Button>
        </div>
      </div>
    </section>
  );
}
