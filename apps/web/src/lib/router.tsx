import { useCallback, useEffect, useState, type AnchorHTMLAttributes, type MouseEvent } from "react";

// Two pages do not justify a routing dependency: history API plus one hook.

export type Route = "studio" | "models";

const PATHS: Record<Route, string> = { studio: "/", models: "/models" };

function routeFromPath(pathname: string): Route {
  return pathname.replace(/\/+$/, "") === "/models" ? "models" : "studio";
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => routeFromPath(window.location.pathname));
  useEffect(() => {
    const update = () => setRoute(routeFromPath(window.location.pathname));
    window.addEventListener("popstate", update);
    return () => window.removeEventListener("popstate", update);
  }, []);
  return route;
}

export function navigate(route: Route): void {
  if (routeFromPath(window.location.pathname) === route) return;
  window.history.pushState(null, "", PATHS[route]);
  window.dispatchEvent(new PopStateEvent("popstate"));
  window.scrollTo({ top: 0 });
}

type LinkProps = AnchorHTMLAttributes<HTMLAnchorElement> & { to: Route };

export function Link({ to, onClick, ...props }: LinkProps) {
  const handleClick = useCallback(
    (event: MouseEvent<HTMLAnchorElement>) => {
      onClick?.(event);
      if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey) return;
      event.preventDefault();
      navigate(to);
    },
    [onClick, to],
  );
  return <a href={PATHS[to]} onClick={handleClick} {...props} />;
}
