import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { App } from "../App";
import { deferred, fixtures, mockApi } from "../test/mockApi";

// The Dataset V1 train sentence the analyze fixtures were recorded with.
const SORANI = "زۆر جار ناوی ئەو کەسانەم لەبیر دەچێتەوە کە چاوم پێیان دەکەوێت.";

function setup(routes: Parameters<typeof mockApi>[0] = {}) {
  const api = mockApi({
    "GET /api/v1/models": () => ({ body: fixtures.models }),
    "POST /api/v1/analyze": (call) => ({
      body: call.body?.model === "fasttext" ? fixtures.analyzeCkbFasttext : fixtures.analyzeCkbTfidf,
    }),
    "POST /api/v1/detect": () => ({ body: fixtures.detectCkbFasttext }),
    ...routes,
  });
  const user = userEvent.setup();
  render(<App />);
  return { ...api, user };
}

const editor = () => screen.getByLabelText("Text to analyze") as HTMLTextAreaElement;
const analyzeButton = () => screen.getByRole("button", { name: /Analyze/ });
const results = () => screen.getByRole("region", { name: "Results" });

async function ready() {
  await screen.findByRole("option", { name: "TF-IDF + LR" });
}

describe("Studio", () => {
  it("tracks input in code points and enables Analyze only for valid text", async () => {
    const { user } = setup();
    await ready();
    expect(analyzeButton()).toBeDisabled();
    expect(screen.getByText("0 / 5,000")).toBeInTheDocument();

    await user.type(editor(), "   ");
    expect(analyzeButton()).toBeDisabled();

    await user.clear(editor());
    await user.type(editor(), "Ez 👋");
    expect(screen.getByText("4 / 5,000")).toBeInTheDocument();
    expect(analyzeButton()).toBeEnabled();
  });

  it("sends the text unchanged and renders the real model response", async () => {
    const pending = deferred<{ body: unknown }>();
    const { user, calls } = setup({ "POST /api/v1/analyze": () => pending.promise });
    await ready();

    const text = `  ${SORANI}\n`;
    await user.click(editor());
    await user.paste(text);
    await user.click(analyzeButton());

    expect(results()).toHaveAttribute("aria-busy", "true");
    const request = calls.find((call) => call.path === "/api/v1/analyze");
    expect(request?.body).toEqual({
      text,
      task: "language",
      language: "auto",
      model: "tfidf_lr",
      policy: "tuned",
    });

    await act(async () => pending.resolve({ body: fixtures.analyzeCkbTfidf }));
    const panel = within(results());
    expect(await panel.findByRole("heading", { name: "Sorani" })).toBeInTheDocument();
    expect(panel.getByText("Central Kurdish")).toBeInTheDocument();
    expect(panel.getAllByText(">99.9%").length).toBeGreaterThan(0);
    expect(panel.getByText("Calibrated estimate")).toBeInTheDocument();
    expect(panel.getAllByRole("row")).toHaveLength(8); // header + seven languages
    expect(panel.getByTestId("result-model")).toHaveTextContent("TF-IDF + LR");
    expect(results()).toHaveAttribute("aria-busy", "false");
  });

  it("re-runs with the selected model", async () => {
    const { user, calls } = setup();
    await ready();
    await user.click(editor());
    await user.paste(SORANI);
    await user.keyboard("{Meta>}{Enter}{/Meta}");
    await within(results()).findByRole("heading", { name: "Sorani" });

    await user.selectOptions(screen.getByRole("combobox", { name: "Model" }), "fasttext");
    await waitFor(() => expect(screen.getByTestId("result-model")).toHaveTextContent("fastText"));
    const analyzeCalls = calls.filter((call) => call.path === "/api/v1/analyze");
    expect(analyzeCalls.map((call) => call.body?.model)).toEqual(["tfidf_lr", "fasttext"]);
  });

  it("ignores a stale response that arrives after a newer one", async () => {
    const first = deferred<{ body: unknown }>();
    let count = 0;
    const { user } = setup({
      "POST /api/v1/analyze": () => (++count === 1 ? first.promise : { body: fixtures.analyzeCkbFasttext }),
    });
    await ready();
    await user.click(editor());
    await user.paste(SORANI);
    await user.click(analyzeButton());
    await user.selectOptions(screen.getByRole("combobox", { name: "Model" }), "fasttext");
    await user.click(analyzeButton());
    await waitFor(() => expect(screen.getByTestId("result-model")).toHaveTextContent("fastText"));

    await act(async () => first.resolve({ body: fixtures.analyzeCkbTfidf }));
    expect(screen.getByTestId("result-model")).toHaveTextContent("fastText");
  });

  it("shows an inconclusive result without inventing a language", async () => {
    const { user } = setup({ "POST /api/v1/analyze": () => ({ body: fixtures.analyzeInconclusive }) });
    await ready();
    await user.click(editor());
    await user.paste("Tehran Istanbul");
    await user.click(analyzeButton());

    const panel = within(results());
    expect(await panel.findByRole("heading", { name: "Inconclusive" })).toBeInTheDocument();
    expect(panel.getByText(fixtures.analyzeInconclusive.detection.reason_message!)).toBeInTheDocument();
    expect(panel.getByText("Closest candidates")).toBeInTheDocument();
    await user.click(panel.getByRole("button", { name: "Choose language manually" }));
    expect(screen.getByRole("combobox", { name: "Language" })).toHaveFocus();
  });

  it("explains when the server cannot be reached", async () => {
    const { user } = setup({ "POST /api/v1/analyze": () => new TypeError("Failed to fetch") });
    await ready();
    await user.click(editor());
    await user.paste(SORANI);
    await user.click(analyzeButton());
    const alert = await within(results()).findByRole("alert");
    expect(alert).toHaveTextContent("Can't reach the analysis server.");
    expect(within(alert).getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("marks NER as unavailable and does not send a request", async () => {
    const { user, calls } = setup();
    await ready();
    await user.click(editor());
    await user.paste(SORANI);
    await user.click(screen.getByRole("radio", { name: "Entities" }));

    expect(within(results()).getByText("NER model is not installed.")).toBeInTheDocument();
    expect(analyzeButton()).toBeDisabled();
    await user.keyboard("{Meta>}{Enter}{/Meta}");
    expect(calls.some((call) => call.path === "/api/v1/analyze")).toBe(false);

    await user.click(within(results()).getByRole("button", { name: "Detect language instead" }));
    expect(screen.getByRole("radio", { name: "Language" })).toBeChecked();
  });

  it("renders the server's unavailable error if capabilities were stale", async () => {
    const models = structuredClone(fixtures.models);
    models.tasks = models.tasks.map((task) => ({ ...task, available: true, message: null }));
    const { user } = setup({
      "GET /api/v1/models": () => ({ body: models }),
      "POST /api/v1/analyze": () => ({ status: 503, body: fixtures.taskUnavailable }),
    });
    await ready();
    await user.click(editor());
    await user.paste(SORANI);
    await user.click(screen.getByRole("radio", { name: "Entities" }));
    await user.click(analyzeButton());
    expect(await within(results()).findByText("NER model is not installed.")).toBeInTheDocument();
  });

  it("handles right-to-left and mixed-direction input", async () => {
    const { user } = setup();
    await ready();
    expect(editor()).toHaveAttribute("dir", "auto");
    await user.click(editor());
    await user.paste(`${SORANI}\nKurdish NLP`);
    expect(editor().value).toBe(`${SORANI}\nKurdish NLP`);

    await user.click(screen.getByRole("button", { name: /Text direction/ }));
    await user.click(screen.getByRole("radio", { name: "Right to left" }));
    expect(editor()).toHaveAttribute("dir", "rtl");
    expect(editor().value).toBe(`${SORANI}\nKurdish NLP`);
  });

  it("compares with the other model on request", async () => {
    const { user, calls } = setup();
    await ready();
    await user.click(editor());
    await user.paste(SORANI);
    await user.click(analyzeButton());
    await within(results()).findByRole("heading", { name: "Sorani" });

    await user.click(within(results()).getByRole("button", { name: "Compare" }));
    expect(await within(results()).findByText("Agrees")).toBeInTheDocument();
    const detect = calls.find((call) => call.path === "/api/v1/detect");
    expect(detect?.body).toMatchObject({ model: "fasttext", text: SORANI });
  });

  it("keeps the draft and settings across a reload of the tab", async () => {
    const { user } = setup();
    await ready();
    await user.click(editor());
    await user.paste("Ez diçim malê");
    await user.selectOptions(screen.getByRole("combobox", { name: "Model" }), "fasttext");
    cleanup();

    render(<App />);
    await ready();
    expect(editor().value).toBe("Ez diçim malê");
    expect(screen.getByRole("combobox", { name: "Model" })).toHaveValue("fasttext");
  });
});
