import { Code } from "../../components/StatusBadge";
import { cx } from "../../lib/cx";
import { formatPercent } from "../../lib/format";
import type { ClassScore } from "../../lib/types";

interface ScoreListProps {
  scores: ClassScore[];
  highlight: string | null;
  primaryLabel: string;
  compare?: { label: string; scores: ClassScore[] } | null;
}

function Bar({ value, tone, thin }: { value: number; tone: "accent" | "muted" | "compare"; thin?: boolean }) {
  // A visible sliver for any non-negligible score; zero stays empty.
  const width = value > 0.0005 ? Math.max(value * 100, 1) : 0;
  return (
    <span className={cx("block overflow-hidden rounded-full bg-track", thin ? "h-1" : "h-1.5")}>
      <span
        className={cx(
          "block h-full rounded-full transition-[width] duration-200 ease-out-soft",
          tone === "accent" && "bg-accent",
          tone === "muted" && "bg-ink-3/45",
          tone === "compare" && "bg-compare",
        )}
        style={{ width: `${width}%` }}
      />
    </span>
  );
}

export function ScoreList({ scores, highlight, primaryLabel, compare }: ScoreListProps) {
  const compareScores = new Map(compare?.scores.map((item) => [item.code, item.score]));
  return (
    <div>
      {compare && (
        <div aria-hidden className="mb-2 flex items-center gap-4 text-[11.5px] text-ink-3">
          <span className="inline-flex items-center gap-1.5">
            <span className="h-1.5 w-3 rounded-full bg-accent" />
            {primaryLabel}
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-1 w-3 rounded-full bg-compare" />
            {compare.label}
          </span>
        </div>
      )}
      <table className="w-full table-fixed border-collapse text-left">
        <caption className="sr-only">Per-language scores, highest first</caption>
        <colgroup>
          <col className={compare ? "w-[44%]" : "w-[47%]"} />
          <col />
          <col className={compare ? "w-16" : "w-[4.5rem]"} />
          {compare && <col className="w-[3.75rem]" />}
        </colgroup>
        <thead className="sr-only">
          <tr>
            <th scope="col">Language</th>
            <th scope="col">Score bar</th>
            <th scope="col">{primaryLabel}</th>
            {compare && <th scope="col">{compare.label}</th>}
          </tr>
        </thead>
        <tbody>
          {scores.map((item) => {
            const top = item.code === highlight;
            const other = compareScores.get(item.code);
            return (
              <tr key={item.code} className="align-middle">
                <th scope="row" className="py-[5px] pr-3 font-normal">
                  <span className="flex min-w-0 items-baseline gap-1.5">
                    <span className={cx("truncate text-[13px]", top ? "font-medium text-ink" : "text-ink-2")}>
                      {item.name}
                    </span>
                    <Code className="text-[11px]">{item.code}</Code>
                  </span>
                </th>
                <td className="py-[5px]">
                  <span className="flex flex-col gap-[3px]">
                    <Bar value={item.score} tone={top ? "accent" : "muted"} />
                    {compare && <Bar value={other ?? 0} tone="compare" thin />}
                  </span>
                </td>
                <td className="tabular py-[5px] pl-3 text-right text-[12.5px] leading-tight">
                  <span className={top ? "font-medium text-ink" : "text-ink-2"}>{formatPercent(item.score)}</span>
                </td>
                {compare && (
                  <td className="tabular py-[5px] text-right text-[11.5px] leading-tight text-ink-3">
                    {other === undefined ? "—" : formatPercent(other)}
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
