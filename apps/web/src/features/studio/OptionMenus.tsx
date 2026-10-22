import { Check, PilcrowLeft, PilcrowRight, SlidersHorizontal } from "lucide-react";

import { IconButton } from "../../components/Button";
import { Popover } from "../../components/Popover";
import { cx } from "../../lib/cx";
import type { PolicyInfo, PolicyName } from "../../lib/types";

export type Direction = "auto" | "rtl" | "ltr";

interface ChoiceProps<T extends string> {
  name: string;
  value: T;
  options: { value: T; label: string; hint?: string }[];
  onChange: (value: T) => void;
  onDone: () => void;
}

function ChoiceList<T extends string>({ name, value, options, onChange, onDone }: ChoiceProps<T>) {
  return (
    <fieldset>
      <legend className="px-2 pt-1.5 pb-1 text-[11.5px] font-medium text-ink-3">{name}</legend>
      {options.map((option) => {
        const checked = option.value === value;
        return (
          <label
            key={option.value}
            className={cx(
              "flex cursor-pointer items-start gap-2.5 rounded-lg px-2 py-1.5 text-[13px] transition-colors duration-150",
              "hover:bg-wash has-[:focus-visible]:bg-wash",
            )}
          >
            <input
              type="radio"
              name={name}
              value={option.value}
              checked={checked}
              onChange={() => onChange(option.value)}
              onKeyDown={(event) => event.key === "Enter" && onDone()}
              onClick={(event) => event.detail > 0 && onDone()}
              className="peer sr-only"
            />
            <Check
              aria-hidden
              className={cx("mt-0.5 size-3.5 shrink-0 text-accent", checked ? "opacity-100" : "opacity-0")}
              strokeWidth={2.25}
            />
            <span className="min-w-0">
              <span className={cx("block", checked ? "font-medium text-ink" : "text-ink-2")}>{option.label}</span>
              {option.hint && <span className="block text-[12px] leading-snug text-ink-3">{option.hint}</span>}
            </span>
          </label>
        );
      })}
    </fieldset>
  );
}

export function DirectionMenu({
  value,
  effective,
  onChange,
}: {
  value: Direction;
  effective: "rtl" | "ltr";
  onChange: (value: Direction) => void;
}) {
  const Icon = effective === "rtl" ? PilcrowLeft : PilcrowRight;
  const label = value === "auto" ? "Auto" : value.toUpperCase();
  return (
    <Popover
      align="end"
      className="w-56"
      trigger={({ toggle, ref, ...aria }) => (
        <IconButton
          ref={ref}
          onClick={toggle}
          label={`Text direction: ${value === "auto" ? "automatic" : value.toUpperCase()}`}
          tip="Text direction"
          className="w-auto gap-1 px-2"
          {...aria}
        >
          <Icon aria-hidden className="size-4" strokeWidth={1.75} />
          <span className="text-[12px] font-medium max-[400px]:sr-only">{label}</span>
        </IconButton>
      )}
    >
      {(close) => (
        <ChoiceList<Direction>
          name="Text direction"
          value={value}
          onChange={onChange}
          onDone={close}
          options={[
            { value: "auto", label: "Automatic", hint: "Each paragraph follows its script." },
            { value: "rtl", label: "Right to left" },
            { value: "ltr", label: "Left to right" },
          ]}
        />
      )}
    </Popover>
  );
}

function referenceHint(policy: PolicyInfo | undefined): string {
  if (!policy) return "Raw scores with the reference thresholds.";
  return `Raw scores. Abstains below ${policy.confidence_threshold.toFixed(2)} confidence or a ${policy.margin_threshold.toFixed(2)} margin.`;
}

export function AdvancedMenu({
  policy,
  policies,
  onPolicyChange,
}: {
  policy: PolicyName;
  policies: PolicyInfo[];
  onPolicyChange: (value: PolicyName) => void;
}) {
  return (
    <Popover
      side="top"
      className="w-[min(18rem,calc(100vw-2rem))]"
      trigger={({ toggle, ref, ...aria }) => (
        <IconButton
          ref={ref}
          onClick={toggle}
          label="Detection options"
          className={policy !== "tuned" ? "text-accent" : undefined}
          {...aria}
        >
          <SlidersHorizontal aria-hidden className="size-4" strokeWidth={1.75} />
        </IconButton>
      )}
    >
      {(close) => (
        <ChoiceList<PolicyName>
          name="Abstention policy"
          value={policy}
          onChange={onPolicyChange}
          onDone={close}
          options={[
            {
              value: "tuned",
              label: "Tuned",
              hint: "Calibrated scores and the frozen policy fitted on held-out dev data.",
            },
            {
              value: "reference",
              label: "Reference",
              hint: referenceHint(policies.find((item) => item.name === "reference")),
            },
          ]}
        />
      )}
    </Popover>
  );
}
