import { useEffect } from "react";

import { Header } from "./components/Header";
import { useCapabilities } from "./hooks/useCapabilities";
import { useRoute } from "./lib/router";
import { ModelsPage } from "./pages/ModelsPage";
import { StudioPage } from "./pages/StudioPage";

const TITLES = { studio: "Kurdish NLP Studio", models: "Models · Kurdish NLP Studio" } as const;

export function App() {
  const route = useRoute();
  const capabilities = useCapabilities();

  useEffect(() => {
    document.title = TITLES[route];
  }, [route]);

  return (
    <div className="min-h-dvh">
      <Header route={route} />
      <main className="mx-auto max-w-[1160px] px-4 pt-8 pb-16 sm:px-6 sm:pt-10 lg:px-8">
        {route === "studio" ? <StudioPage capabilities={capabilities} /> : <ModelsPage capabilities={capabilities} />}
      </main>
    </div>
  );
}
