import { useState } from "react";

import { cx } from "../../lib/cx";
import type { GrammarResult as GrammarData } from "../../lib/types";

export function GrammarResult({ result }: { result: GrammarData }) {
  const [selected, setSelected] = useState<number | null>(null);
  const tokens = result.tokens;
  const active = selected === null ? null : tokens.find((token) => token.index === selected);

  return (
    <div className="animate-[fade-in_160ms_var(--ease-out-soft)]">
      <p className="text-[12.5px] font-medium text-ink-3">
        Grammar · <span className="tabular">{tokens.length}</span> tokens
      </p>

      <div className="mt-3 -mx-1 overflow-x-auto px-1">
        <table className="w-full min-w-[22rem] text-left text-[12.5px]">
          <caption className="sr-only">Tokens with lemma, part of speech and dependency</caption>
          <thead>
            <tr className="text-[11.5px] text-ink-3">
              <th scope="col" className="py-2 pr-2 font-medium">#</th>
              <th scope="col" className="py-2 pr-2 font-medium">Token</th>
              <th scope="col" className="py-2 pr-2 font-medium">Lemma</th>
              <th scope="col" className="py-2 pr-2 font-medium">POS</th>
              <th scope="col" className="py-2 font-medium">Dependency</th>
            </tr>
          </thead>
          <tbody>
            {tokens.map((token) => (
              <tr
                key={token.index}
                onClick={() => setSelected(token.index)}
                className={cx("cursor-pointer border-t border-line/70", selected === token.index && "bg-wash")}
              >
                <td className="tabular py-1.5 pr-2 font-mono text-[11.5px] text-ink-3">{token.index + 1}</td>
                <td dir="auto" className="py-1.5 pr-2 [unicode-bidi:plaintext]">{token.text}</td>
                <td dir="auto" className="py-1.5 pr-2 text-ink-2 [unicode-bidi:plaintext]">{token.lemma ?? "—"}</td>
                <td className="py-1.5 pr-2 font-mono text-[11.5px]">{token.pos ?? "—"}</td>
                <td className="py-1.5 font-mono text-[11.5px] text-ink-2">
                  {token.dep ?? "—"}
                  {token.dep && token.head !== token.index && <span className="text-ink-3"> → {token.head + 1}</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {active && (
        <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 rounded-lg bg-wash px-3 py-2.5 text-[12.5px]">
          <dt className="text-ink-3">Token</dt>
          <dd dir="auto" className="[unicode-bidi:plaintext]">{active.text}</dd>
          <dt className="text-ink-3">Tag</dt>
          <dd className="font-mono text-[12px]">{active.tag ?? "—"}</dd>
          <dt className="text-ink-3">Morphology</dt>
          <dd className="font-mono text-[12px]">
            {Object.keys(active.morph).length
              ? Object.entries(active.morph)
                  .map(([key, value]) => `${key}=${value}`)
                  .join(" | ")
              : "—"}
          </dd>
          <dt className="text-ink-3">Offsets</dt>
          <dd className="tabular font-mono text-[12px]">
            {active.start}–{active.end}
          </dd>
        </dl>
      )}
      <p className="mt-4 text-[12px] text-ink-3">
        {result.model.name}
        {result.model.version ? ` · v${result.model.version}` : ""}
      </p>
    </div>
  );
}
