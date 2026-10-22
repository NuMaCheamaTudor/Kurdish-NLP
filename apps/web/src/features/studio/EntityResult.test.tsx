import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type { EntitiesResult } from "../../lib/types";
import { EntityResult } from "./EntityResult";

// Shape test only: no NER model is installed, so this is hand-written input
// for the renderer, never shown in the app.
const TEXT = "سڵاو 👋 لە هەولێر";
const START = Array.from(TEXT).indexOf("ه");
const RESULT: EntitiesResult = {
  model: { id: "test", name: "Test pipeline", version: null },
  labels: ["LOC", "PER"],
  entities: [{ text: "هەولێر", label: "LOC", start: START, end: START + 6 }],
};

describe("EntityResult", () => {
  it("highlights entities by code point offsets, even after astral characters", async () => {
    render(<EntityResult text={TEXT} result={RESULT} />);
    const mark = screen.getByRole("button", { name: "هەولێر, LOC" });
    expect(mark).toHaveTextContent("هەولێر");
    expect(mark.parentElement).toHaveAttribute("dir", "auto");

    await userEvent.click(mark);
    const details = screen.getByText("Offsets").closest("dl")!;
    expect(within(details).getByText(`${START}–${START + 6}`)).toBeInTheDocument();
    expect(screen.getByRole("list", { name: "Entity labels" })).toHaveTextContent("LOC 1");
  });
});
