import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { App } from "../App";
import { fixtures, mockApi } from "../test/mockApi";

describe("Models page", () => {
  it("shows measured metrics labelled as internal, and unavailable pipelines", async () => {
    window.history.replaceState(null, "", "/models");
    mockApi({ "GET /api/v1/models": () => ({ body: fixtures.models }) });
    render(<App />);

    const tfidf = (await screen.findByRole("heading", { name: "TF-IDF + Logistic Regression" })).closest("article")!;
    const card = within(tfidf);
    expect(card.getByText("Default")).toBeInTheDocument();
    expect(card.getByText("Ready")).toBeInTheDocument();
    expect(card.getByText("99.65%")).toBeInTheDocument();
    expect(card.getAllByText("Internal test")).toHaveLength(2);
    expect(card.getByText("0 / 217")).toBeInTheDocument();
    expect(screen.getByText(/It is not a measure of real-world accuracy/)).toBeInTheDocument();

    const ner = screen.getByRole("heading", { name: "Sorani NER" }).closest("li")!;
    expect(within(ner).getByText("Not installed")).toBeInTheDocument();
    expect(within(ner).queryByText(/%/)).not.toBeInTheDocument();
  });

  it.each([
    ["a refused connection", () => new TypeError("Failed to fetch")],
    ["a proxy 502 without an API error body", () => ({ status: 502, body: null })],
  ])("reports an unreachable server on %s", async (_, reply) => {
    window.history.replaceState(null, "", "/models");
    mockApi({ "GET /api/v1/models": reply });
    render(<App />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Can't reach the analysis server.");
  });
});
