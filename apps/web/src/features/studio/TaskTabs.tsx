import { useRef, type KeyboardEvent } from "react";

import { cx } from "../../lib/cx";
import type { Task, TaskInfo } from "../../lib/types";

const TASKS: { id: Task; label: string }[] = [
  { id: "language", label: "Language" },
  { id: "entities", label: "Entities" },
  { id: "grammar", label: "Grammar" },
];

interface TaskTabsProps {
  value: Task;
  tasks: TaskInfo[] | undefined;
  onChange: (task: Task) => void;
}

/** A radio group: one tab stop, arrow keys move the selection. */
export function TaskTabs({ value, tasks, onChange }: TaskTabsProps) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
    if (!step) return;
    event.preventDefault();
    const index = TASKS.findIndex((task) => task.id === value);
    const next = TASKS[(index + step + TASKS.length) % TASKS.length]!;
    onChange(next.id);
    refs.current[TASKS.indexOf(next)]?.focus();
  };

  return (
    <div role="radiogroup" aria-label="Task" className="flex items-center gap-0.5" onKeyDown={onKeyDown}>
      {TASKS.map((task, index) => {
        const info = tasks?.find((item) => item.id === task.id);
        const unavailable = info ? !info.available : false;
        const selected = task.id === value;
        return (
          <button
            key={task.id}
            ref={(node) => {
              refs.current[index] = node;
            }}
            type="button"
            role="radio"
            aria-checked={selected}
            aria-description={unavailable ? "Not installed" : undefined}
            tabIndex={selected ? 0 : -1}
            onClick={() => onChange(task.id)}
            className={cx(
              "relative flex h-11 items-center gap-1.5 px-2 text-[13.5px] transition-colors duration-150 sm:px-2.5",
              selected ? "font-medium text-ink" : "text-ink-3 hover:text-ink",
              "after:absolute after:inset-x-2 after:-bottom-px after:h-0.5 after:rounded-full after:transition-colors after:duration-150",
              selected ? "after:bg-ink" : "after:bg-transparent",
            )}
          >
            {task.label}
            {unavailable && (
              <span
                aria-hidden
                title="Not installed"
                className="size-1.5 rounded-full border border-current opacity-60"
              />
            )}
          </button>
        );
      })}
    </div>
  );
}
