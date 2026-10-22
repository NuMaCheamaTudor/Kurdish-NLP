import { useMemo, useState } from "react";

import { Code } from "../../components/StatusBadge";
import { cx } from "../../lib/cx";
import { segmentByCodePoints } from "../../lib/text";
import type { EntitiesResult, Entity } from "../../lib/types";

// Muted label hues; text stays dark enough for AA contrast on each background.
const HUES = [
  "bg-[#e3efec] text-[#1f4d4a] ring-[#b9d6d0]",
  "bg-[#f3eadb] text-[#5c4113] ring-[#e3cfa8]",
  "bg-[#e6ebf3] text-[#2c4363] ring-[#c3cfe2]",
  "bg-[#f1e4e6] text-[#5f2e36] ring-[#e0c0c6]",
  "bg-[#ebe8f1] text-[#433a5e] ring-[#cfc8e0]",
  "bg-[#e7eedf] text-[#3c4f24] ring-[#c8d8b4]",
];

function hueFor(label: string, labels: string[]): string {
  const index = labels.indexOf(label);
  return HUES[(index < 0 ? 0 : index) % HUES.length]!;
}

export function EntityResult({ text, result }: { text: string; result: EntitiesResult }) {
  const [selected, setSelected] = useState<number | null>(null);
  const labels = result.labels.length ? result.labels : [...new Set(result.entities.map((e) => e.label))];
  const segments = useMemo(
    () =>
      segmentByCodePoints<{ entity: Entity; index: number }>(
        text,
        result.entities.map((entity, index) => ({ start: entity.start, end: entity.end, value: { entity, index } })),
      ),
    [text, result.entities],
  );
  const counts = new Map<string, number>();
  for (const entity of result.entities) counts.set(entity.label, (counts.get(entity.label) ?? 0) + 1);
  const active = selected === null ? null : (result.entities[selected] ?? null);

  return (
    <div className="animate-[fade-in_160ms_var(--ease-out-soft)]">
      <p className="text-[12.5px] font-medium text-ink-3">
        Named entities · <span className="tabular">{result.entities.length}</span>
      </p>

      <p dir="auto" className="user-text mt-3 text-ink">
        {segments.map((segment, i) =>
          segment.span ? (
            <button
              key={i}
              type="button"
              onClick={() => setSelected(segment.span!.value.index)}
              aria-pressed={selected === segment.span.value.index}
              aria-label={`${segment.text}, ${segment.span.value.entity.label}`}
              className={cx(
                "rounded-[5px] px-0.5 ring-1 ring-inset transition-shadow duration-150 [unicode-bidi:isolate]",
                hueFor(segment.span.value.entity.label, labels),
                selected === segment.span.value.index && "ring-2",
              )}
            >
              {segment.text}
            </button>
          ) : (
            <span key={i}>{segment.text}</span>
          ),
        )}
      </p>

      {counts.size > 0 && (
        <ul aria-label="Entity labels" className="mt-4 flex flex-wrap gap-1.5">
          {[...counts].map(([label, count]) => (
            <li key={label} className={cx("rounded-md px-1.5 py-0.5 text-[11.5px] font-medium ring-1 ring-inset", hueFor(label, labels))}>
              {label} <span className="tabular opacity-70">{count}</span>
            </li>
          ))}
        </ul>
      )}

      {active && (
        <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 rounded-lg bg-wash px-3 py-2.5 text-[12.5px]">
          <dt className="text-ink-3">Text</dt>
          <dd dir="auto" className="[unicode-bidi:plaintext]">{active.text}</dd>
          <dt className="text-ink-3">Label</dt>
          <dd>{active.label}</dd>
          <dt className="text-ink-3">Offsets</dt>
          <dd className="tabular font-mono text-[12px]">
            {active.start}–{active.end}
          </dd>
        </dl>
      )}

      <table className="mt-5 w-full border-t border-line text-left text-[12.5px]">
        <caption className="sr-only">Entities with code point offsets</caption>
        <thead>
          <tr className="text-[11.5px] text-ink-3">
            <th scope="col" className="py-2 font-medium">Entity</th>
            <th scope="col" className="py-2 font-medium">Label</th>
            <th scope="col" className="py-2 text-right font-medium">Start–end</th>
          </tr>
        </thead>
        <tbody>
          {result.entities.map((entity, index) => (
            <tr
              key={`${entity.start}-${entity.end}`}
              onClick={() => setSelected(index)}
              className={cx("cursor-pointer border-t border-line/70", selected === index && "bg-wash")}
            >
              <td dir="auto" className="py-1.5 pr-2 [unicode-bidi:plaintext]">{entity.text}</td>
              <td className="py-1.5 pr-2">{entity.label}</td>
              <td className="py-1.5 text-right">
                <Code>
                  {entity.start}–{entity.end}
                </Code>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {result.entities.length === 0 && <p className="mt-3 text-[13px] text-ink-3">No entities found.</p>}
      <p className="mt-4 text-[12px] text-ink-3">
        {result.model.name}
        {result.model.version ? ` · v${result.model.version}` : ""}
      </p>
    </div>
  );
}
