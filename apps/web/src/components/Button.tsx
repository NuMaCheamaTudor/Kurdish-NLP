import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from "react";

import { cx } from "../lib/cx";

type Variant = "primary" | "secondary" | "ghost";

const VARIANTS: Record<Variant, string> = {
  primary:
    "bg-accent text-white hover:bg-accent-strong disabled:bg-line-strong disabled:text-white/90 shadow-[inset_0_1px_0_rgb(255_255_255/0.12)]",
  secondary:
    "border border-line bg-surface text-ink hover:border-line-strong hover:bg-wash disabled:text-ink-3",
  ghost: "text-ink-2 hover:bg-wash hover:text-ink disabled:text-ink-3",
};

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: "sm" | "md";
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "secondary", size = "md", className, type = "button", ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cx(
        "inline-flex shrink-0 items-center justify-center gap-1.5 rounded-lg font-medium whitespace-nowrap",
        "transition-[background-color,border-color,color] duration-150 ease-out-soft disabled:cursor-not-allowed",
        size === "md" ? "h-8 px-3 text-[13px]" : "h-7 px-2.5 text-[12.5px]",
        VARIANTS[variant],
        className,
      )}
      {...props}
    />
  );
});

interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  label: string;
  tip?: ReactNode;
  tipAlign?: "center" | "end";
}

/** Square icon-only button with an accessible name and a hover/focus tip. */
export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(function IconButton(
  { label, tip, tipAlign = "center", className, children, type = "button", ...props },
  ref,
) {
  return (
    <span className="group/tip relative inline-flex">
      <button
        ref={ref}
        type={type}
        aria-label={label}
        className={cx(
          "inline-flex size-8 items-center justify-center rounded-lg text-ink-3",
          "transition-colors duration-150 hover:bg-wash hover:text-ink disabled:pointer-events-none disabled:opacity-40",
          className,
        )}
        {...props}
      >
        {children}
      </button>
      <span
        role="presentation"
        className={cx(
          "pointer-events-none absolute top-full z-30 mt-1.5 rounded-md bg-ink px-2 py-1 text-[11.5px] font-medium whitespace-nowrap text-white",
          "opacity-0 transition-opacity duration-150 group-hover/tip:opacity-100 group-hover/tip:delay-300 max-sm:hidden",
          "group-has-[:focus-visible]/tip:opacity-100",
          tipAlign === "end" ? "right-0" : "left-1/2 -translate-x-1/2",
        )}
      >
        {tip ?? label}
      </span>
    </span>
  );
});
