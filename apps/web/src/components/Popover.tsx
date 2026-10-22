import { useCallback, useEffect, useId, useRef, useState, type ReactNode } from "react";

import { cx } from "../lib/cx";

interface PopoverProps {
  trigger: (props: {
    open: boolean;
    toggle: () => void;
    ref: React.RefObject<HTMLButtonElement | null>;
    "aria-expanded": boolean;
    "aria-controls": string;
  }) => ReactNode;
  children: (close: () => void) => ReactNode;
  align?: "start" | "end";
  side?: "top" | "bottom";
  className?: string;
}

/** Click-to-open panel. Closes on Escape (returning focus) and outside clicks. */
export function Popover({ trigger, children, align = "start", side = "bottom", className }: PopoverProps) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const panelRef = useRef<HTMLDivElement | null>(null);

  const close = useCallback((restoreFocus = true) => {
    setOpen(false);
    if (restoreFocus) triggerRef.current?.focus();
  }, []);

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        close();
      }
    };
    const onPointer = (event: PointerEvent) => {
      const target = event.target as Node;
      if (!panelRef.current?.contains(target) && !triggerRef.current?.contains(target)) close(false);
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onPointer);
    panelRef.current?.querySelector<HTMLElement>("input:checked, button, input")?.focus();
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("pointerdown", onPointer);
    };
  }, [open, close]);

  return (
    <div className="relative inline-flex">
      {trigger({
        open,
        toggle: () => setOpen((value) => !value),
        ref: triggerRef,
        "aria-expanded": open,
        "aria-controls": id,
      })}
      {open && (
        <div
          ref={panelRef}
          id={id}
          className={cx(
            "absolute z-40 rounded-xl border border-line bg-surface p-1.5 shadow-pop",
            "origin-top animate-[pop-in_140ms_var(--ease-out-soft)]",
            side === "bottom" ? "top-full mt-1.5" : "bottom-full mb-1.5",
            align === "end" ? "right-0" : "left-0",
            className,
          )}
        >
          {children(() => close())}
        </div>
      )}
    </div>
  );
}
