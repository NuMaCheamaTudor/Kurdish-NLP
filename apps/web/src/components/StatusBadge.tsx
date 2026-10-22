import { cx } from "../lib/cx";

type Tone = "ready" | "muted" | "accent";

const TONES: Record<Tone, string> = {
  ready: "text-accent-strong",
  muted: "text-ink-3",
  accent: "text-accent-strong",
};

const DOTS: Record<Tone, string> = {
  ready: "bg-accent",
  muted: "border border-ink-3/60 bg-transparent",
  accent: "bg-accent",
};

export function StatusBadge({ tone, children }: { tone: Tone; children: React.ReactNode }) {
  return (
    <span className={cx("inline-flex items-center gap-1.5 text-[12px] font-medium", TONES[tone])}>
      <span aria-hidden className={cx("size-1.5 rounded-full", DOTS[tone])} />
      {children}
    </span>
  );
}

export function Tag({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <span
      className={cx(
        "inline-flex h-5 items-center rounded-md border border-line bg-wash px-1.5 text-[11px] font-medium text-ink-2",
        className,
      )}
    >
      {children}
    </span>
  );
}

export function Code({ children, className }: { children: React.ReactNode; className?: string }) {
  return <span className={cx("font-mono text-[11.5px] text-ink-3", className)}>{children}</span>;
}
