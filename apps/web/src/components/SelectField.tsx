import { ChevronDown } from "lucide-react";
import type { ReactNode } from "react";

import { cx } from "../lib/cx";

export interface SelectOption {
  value: string;
  label: string;
  disabled?: boolean;
}

interface SelectFieldProps {
  id?: string;
  label: string;
  value: string;
  options: SelectOption[];
  onChange: (value: string) => void;
  disabled?: boolean;
  icon?: ReactNode;
  className?: string;
}

/**
 * A compact "Label · Value" control. The visible part is decorative; a native
 * <select> sits on top of it, so keyboard, screen reader and mobile pickers
 * behave natively.
 */
export function SelectField({ id, label, value, options, onChange, disabled, icon, className }: SelectFieldProps) {
  const current = options.find((option) => option.value === value);
  return (
    <label
      className={cx(
        "relative inline-flex h-8 min-w-0 items-center gap-1.5 rounded-lg border border-line bg-surface pr-2 pl-2.5 text-[13px]",
        "transition-colors duration-150 has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-accent",
        disabled ? "opacity-60" : "hover:border-line-strong hover:bg-wash",
        className,
      )}
    >
      {icon && <span className="text-ink-3">{icon}</span>}
      <span className="text-ink-3 max-sm:hidden">{label}</span>
      <span aria-hidden className="min-w-0 truncate font-medium text-ink">
        {current?.label ?? "—"}
      </span>
      <ChevronDown aria-hidden className="size-3.5 shrink-0 text-ink-3" strokeWidth={2} />
      <select
        id={id}
        aria-label={label}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
        className="absolute inset-0 appearance-none opacity-0 focus-visible:outline-none disabled:cursor-not-allowed"
      >
        {options.map((option) => (
          <option key={option.value} value={option.value} disabled={option.disabled}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}
