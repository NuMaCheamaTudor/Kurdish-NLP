import { useEffect, useState } from "react";

/** True only once `active` has stayed true for `delay` ms, so fast requests never flash a spinner. */
export function useDelayedFlag(active: boolean, delay = 250): boolean {
  const [shown, setShown] = useState(false);
  useEffect(() => {
    if (!active) {
      setShown(false);
      return;
    }
    const timer = window.setTimeout(() => setShown(true), delay);
    return () => window.clearTimeout(timer);
  }, [active, delay]);
  return shown;
}
