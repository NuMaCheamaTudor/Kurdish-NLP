import { ArrowUpRight, RotateCcw, WifiOff } from "lucide-react";

import { Button } from "../components/Button";
import { StatusBadge, Tag } from "../components/StatusBadge";
import type { Capabilities } from "../hooks/useCapabilities";
import { useDelayedFlag } from "../hooks/useDelayedFlag";
import { formatBytes, formatMetric, formatNumber } from "../lib/format";
import type { LanguageDetail, ModelInfo, ModelsResponse } from "../lib/types";

const TASK_LABELS = { language: "Language ID", entities: "Named entities", grammar: "Grammar" } as const;

function languageNames(codes: string[], languages: LanguageDetail[]): string {
  return codes.map((code) => languages.find((item) => item.code === code)?.name ?? code).join(", ");
}

export function ModelsPage({ capabilities }: { capabilities: Capabilities }) {
  const data = capabilities.data;
  const showLoading = useDelayedFlag(capabilities.status === "loading" && !data, 300);

  return (
    <div>
      <div className="mb-8 flex flex-wrap items-end justify-between gap-x-6 gap-y-3">
        <div>
          <h1 className="text-[22px] leading-tight font-semibold tracking-[-0.02em]">Models</h1>
          <p className="mt-1 text-[14px] text-ink-3">Classifiers and pipelines on this server.</p>
        </div>
        {data && (
          <a
            href={data.documentation_url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 rounded-md text-[13.5px] font-medium text-accent-strong transition-colors duration-150 hover:text-ink"
          >
            Technical documentation
            <ArrowUpRight aria-hidden className="size-3.5" />
          </a>
        )}
      </div>

      {capabilities.status === "error" && !data && (
        <div role="alert" className="rounded-xl border border-line bg-surface p-5 shadow-card">
          <p className="flex items-center gap-1.5 text-[12.5px] font-medium text-danger">
            <WifiOff aria-hidden className="size-3.5" strokeWidth={2} />
            Server unavailable
          </p>
          <p className="mt-3 text-[14px]">{capabilities.error?.message}</p>
          <Button size="sm" className="mt-4" onClick={capabilities.retry}>
            <RotateCcw aria-hidden className="size-3.5" />
            Retry
          </Button>
        </div>
      )}
      {showLoading && <p className="text-[13px] text-ink-3">Loading models…</p>}
      {data && <ModelSections data={data} />}
    </div>
  );
}

function ModelSections({ data }: { data: ModelsResponse }) {
  const langid = data.models.filter((model) => model.task === "language");
  const tasks = data.models.filter((model) => model.task !== "language");
  const internal = langid.find((model) => model.internal_test)?.internal_test;
  const external = langid.find((model) => model.external_benchmarks.length)?.external_benchmarks[0];

  return (
    <>
      <section aria-labelledby="langid-heading">
        <h2 id="langid-heading" className="mb-3 text-[13px] font-medium text-ink-3">
          Language identification
        </h2>
        <div className="grid gap-4 md:grid-cols-2 md:gap-y-0">
          {langid.map((model) => (
            <LanguageModelCard key={model.id} model={model} languages={data.languages} />
          ))}
        </div>
        {(internal || external) && (
          <p className="mt-4 max-w-[72ch] text-[12.5px] leading-relaxed text-ink-3">
            {internal && (
              <>
                Internal test: {internal.records ? `${formatNumber(internal.records)} ` : ""}held-out records
                from {internal.dataset}, drawn from the same sources as the training data. It is not a
                measure of real-world accuracy.{" "}
              </>
            )}
            {external && (
              <>
                Garrusi recall counts correct predictions on{" "}
                {external.records ? `${formatNumber(external.records)} sentences from ` : ""}
                {external.name}. {external.description}
              </>
            )}
          </p>
        )}
      </section>

      <section aria-labelledby="tasks-heading" className="mt-10">
        <h2 id="tasks-heading" className="mb-3 text-[13px] font-medium text-ink-3">
          Linguistic analysis
        </h2>
        <ul className="divide-y divide-line rounded-xl border border-line bg-surface shadow-card">
          {tasks.map((model) => (
            <TaskModelRow key={model.id} model={model} languages={data.languages} />
          ))}
        </ul>
      </section>
    </>
  );
}

function LanguageModelCard({ model, languages }: { model: ModelInfo; languages: LanguageDetail[] }) {
  const ready = model.status === "ready";
  const garrusi = model.external_benchmarks[0];
  return (
    // Four subgrid rows (header, description, figures, details) keep side-by-side cards aligned.
    <article className="flex flex-col rounded-xl border border-line bg-surface p-5 shadow-card md:row-span-4 md:grid md:grid-rows-subgrid">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-[16px] font-semibold tracking-[-0.01em]">{model.full_name}</h3>
          <p className="mt-0.5 text-[12.5px] text-ink-3">
            {[model.framework, model.version && `v${model.version}`].filter(Boolean).join(" · ")}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2.5">
          {model.default && <Tag>Default</Tag>}
          <StatusBadge tone={ready ? "ready" : "muted"}>{ready ? "Ready" : "Unavailable"}</StatusBadge>
        </div>
      </div>
      <p className="mt-3 text-[13.5px] leading-relaxed text-ink-2">{model.description}</p>

      {ready && model.internal_test ? (
        <dl className="mt-5 grid grid-cols-2 gap-px self-start overflow-hidden rounded-lg border border-line bg-line">
          <Stat label="Macro-F1" note="Internal test" value={formatMetric(model.internal_test.macro_f1)} />
          <Stat
            label="Accuracy"
            note="Internal test"
            value={model.internal_test.accuracy === null ? "—" : formatMetric(model.internal_test.accuracy)}
          />
          {garrusi && (
            <Stat
              label="Garrusi recall"
              note="Southern Kurdish, Latin script"
              value={
                garrusi.correct !== null && garrusi.records !== null
                  ? `${garrusi.correct} / ${garrusi.records}`
                  : formatMetric(garrusi.recall)
              }
            />
          )}
          <Stat label="Size" note="On disk" value={model.size_bytes ? formatBytes(model.size_bytes) : "—"} />
        </dl>
      ) : (
        <ul className="mt-4 space-y-1 self-start rounded-lg bg-wash px-3 py-2.5 text-[12.5px] text-ink-2">
          {(model.unavailable_reasons.length ? model.unavailable_reasons : ["No evaluation recorded."]).map(
            (reason) => (
              <li key={reason}>{reason}</li>
            ),
          )}
        </ul>
      )}

      <dl className="mt-5 grid grid-cols-[5.5rem_minmax(0,1fr)] content-start gap-x-3 gap-y-1.5 border-t border-line pt-4 text-[12.5px]">
        <dt className="text-ink-3">Languages</dt>
        <dd className="text-ink-2">{languageNames(model.languages, languages)}</dd>
        {model.identifier && (
          <>
            <dt className="text-ink-3">Identifier</dt>
            <dd className="truncate font-mono text-[11.5px] leading-[1.6] text-ink-2" title={model.identifier}>
              {model.identifier}
            </dd>
          </>
        )}
        {model.experiment && (
          <>
            <dt className="text-ink-3">Experiment</dt>
            <dd className="text-ink-2">
              {model.experiment}
              {model.temperature !== null && (
                <span className="text-ink-3"> · temperature {model.temperature.toFixed(3)}</span>
              )}
            </dd>
          </>
        )}
        {model.artifact_sha256 && (
          <>
            <dt className="text-ink-3">SHA-256</dt>
            <dd className="truncate font-mono text-[11.5px] leading-[1.6] text-ink-2" title={model.artifact_sha256}>
              {model.artifact_sha256.slice(0, 16)}…
            </dd>
          </>
        )}
      </dl>
    </article>
  );
}

function Stat({ label, note, value }: { label: string; note: string; value: string }) {
  return (
    <div className="bg-surface px-3.5 py-3">
      <dt className="text-[12px] text-ink-3">
        {label}
        <span className="block text-[11px] text-ink-3/80">{note}</span>
      </dt>
      <dd className="tabular mt-1.5 text-[17px] font-semibold tracking-[-0.01em]">{value}</dd>
    </div>
  );
}

function TaskModelRow({ model, languages }: { model: ModelInfo; languages: LanguageDetail[] }) {
  const ready = model.status === "ready";
  return (
    <li className="flex flex-col gap-3 p-5 sm:flex-row sm:items-start sm:justify-between sm:gap-8">
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <h3 className="text-[14.5px] font-semibold">{model.name}</h3>
          <StatusBadge tone={ready ? "ready" : "muted"}>{ready ? "Ready" : "Not installed"}</StatusBadge>
        </div>
        <p className="mt-1 text-[13px] leading-relaxed text-ink-2">{model.description}</p>
        <p className="mt-1.5 text-[12px] text-ink-3">
          {TASK_LABELS[model.task]} · {languageNames(model.languages, languages)} · {model.framework}
        </p>
      </div>
      {!ready && model.unavailable_reasons.length > 0 && (
        <ul className="shrink-0 space-y-0.5 text-[12px] text-ink-3 sm:max-w-[18rem] sm:text-right">
          {model.unavailable_reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
      )}
    </li>
  );
}
