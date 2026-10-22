import { ArrowUpRight } from "lucide-react";

import { cx } from "../lib/cx";
import { Link, type Route } from "../lib/router";
import { LogoMark } from "./Logo";

const REPOSITORY_URL = "https://github.com/NuMaCheamaTudor/Kurdish-NLP";

const NAV: { route: Route; label: string }[] = [
  { route: "studio", label: "Studio" },
  { route: "models", label: "Models" },
];

export function Header({ route }: { route: Route }) {
  return (
    <header className="sticky top-0 z-20 border-b border-line/80 bg-canvas/85 backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-[1160px] items-center gap-6 px-4 sm:px-6 lg:px-8">
        <Link to="studio" className="flex items-center gap-2.5 rounded-md" aria-label="Kurdish NLP, Studio">
          <LogoMark className="size-[22px]" />
          <span className="text-[15px] font-semibold tracking-[-0.01em] whitespace-nowrap">Kurdish NLP</span>
        </Link>
        <nav aria-label="Main" className="flex items-center gap-1 max-sm:-ml-2">
          {NAV.map((item) => {
            const active = item.route === route;
            return (
              <Link
                key={item.route}
                to={item.route}
                aria-current={active ? "page" : undefined}
                className={cx(
                  "rounded-md px-2.5 py-1.5 text-[13.5px] transition-colors duration-150",
                  active ? "bg-ink/[0.06] font-medium text-ink" : "text-ink-3 hover:text-ink",
                )}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
        <a
          href={REPOSITORY_URL}
          target="_blank"
          rel="noreferrer"
          className="ml-auto inline-flex items-center gap-1 rounded-md text-[13.5px] text-ink-3 transition-colors duration-150 hover:text-ink"
        >
          GitHub
          <ArrowUpRight aria-hidden className="size-3.5" />
        </a>
      </div>
    </header>
  );
}
