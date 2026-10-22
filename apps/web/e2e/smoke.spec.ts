import { expect, test, type Page } from "@playwright/test";

// Runs against the real API with the baseline-v1 models. Inputs are the
// built-in examples (Dataset V1 train split) plus hand-typed text; no held-out
// test, dev or UD benchmark sentences.

const results = (page: Page) => page.getByRole("region", { name: "Results" });
const editor = (page: Page) => page.getByLabel("Text to analyze");

async function runExample(page: Page, label: string) {
  await page.goto("/");
  await page.getByRole("button", { name: label, exact: true }).click();
  await expect(results(page).getByText("Detected language")).toBeVisible();
}

test.describe("Studio", () => {
  test("detects Sorani", async ({ page }) => {
    await runExample(page, "Sorani");
    await expect(results(page).getByRole("heading", { name: "Sorani" })).toBeVisible();
    await expect(results(page).getByText("Central Kurdish")).toBeVisible();
  });

  test("detects Kurmanji", async ({ page }) => {
    await runExample(page, "Kurmanji");
    await expect(results(page).getByRole("heading", { name: "Kurmanji" })).toBeVisible();
  });

  test("detects Southern Kurdish", async ({ page }) => {
    await runExample(page, "Southern Kurdish");
    await expect(results(page).getByRole("heading", { name: "Southern Kurdish" })).toBeVisible();
  });

  test("switches between TF-IDF and fastText", async ({ page }) => {
    await runExample(page, "Sorani");
    await expect(page.getByTestId("result-model")).toHaveText("TF-IDF + LR");
    await page.getByRole("combobox", { name: "Model" }).selectOption("fasttext");
    await expect(page.getByTestId("result-model")).toHaveText("fastText");
    await expect(results(page).getByRole("heading", { name: "Sorani" })).toBeVisible();
    await page.getByRole("combobox", { name: "Model" }).selectOption("tfidf_lr");
    await expect(page.getByTestId("result-model")).toHaveText("TF-IDF + LR");
  });

  test("does not submit empty input", async ({ page, request }) => {
    await page.goto("/");
    const analyze = page.getByRole("button", { name: /Analyze/ });
    await expect(analyze).toBeDisabled();
    await editor(page).fill("   \n  ");
    await expect(analyze).toBeDisabled();
    const response = await request.post("/api/v1/detect", { data: { text: "  " } });
    expect(response.status()).toBe(422);
    expect((await response.json()).error.code).toBe("empty_text");
  });

  test("reports NER as not installed", async ({ page, request }) => {
    await page.goto("/");
    await editor(page).fill("Ez diçim malê");
    await page.getByRole("radio", { name: "Entities" }).click();
    await expect(results(page).getByText("NER model is not installed.")).toBeVisible();
    await expect(page.getByRole("button", { name: /Analyze/ })).toBeDisabled();
    const response = await request.post("/api/v1/analyze", { data: { text: "Ez diçim malê", task: "entities" } });
    expect(response.status()).toBe(503);
  });

  test("keeps the draft and settings after a reload", async ({ page }) => {
    await runExample(page, "Kurmanji");
    await page.getByRole("combobox", { name: "Model" }).selectOption("fasttext");
    await expect(page.getByTestId("result-model")).toHaveText("fastText");
    await page.reload();
    await expect(editor(page)).toHaveValue(/Koma dîlanê/);
    await expect(page.getByRole("combobox", { name: "Model" })).toHaveValue("fasttext");
    await page.keyboard.press("ControlOrMeta+Enter");
    await expect(results(page).getByRole("heading", { name: "Kurmanji" })).toBeVisible();
  });

  test("handles right-to-left and mixed-direction text", async ({ page }) => {
    await page.goto("/");
    const text = "ئەمڕۆ Kurdish NLP تاقی دەکەمەوە (0.1)\nEz ê sibê werim.";
    await editor(page).fill(text);
    await expect(editor(page)).toHaveAttribute("dir", "auto");
    expect(await editor(page).evaluate((el) => getComputedStyle(el).direction)).toBe("rtl");
    expect(await editor(page).evaluate((el) => getComputedStyle(el).unicodeBidi)).toBe("plaintext");
    await page.keyboard.press("ControlOrMeta+Enter");
    await expect(results(page).getByText("Detected language")).toBeVisible();
    await expect(editor(page)).toHaveValue(text);
  });
});

test("fits a phone screen @mobile", async ({ page }) => {
  await runExample(page, "Sorani");
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(0);
  await page.goto("/models");
  await expect(page.getByRole("heading", { name: "Models", level: 1 })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(0);
});

test("models page shows real metadata", async ({ page }) => {
  await page.goto("/models");
  await expect(page.getByRole("heading", { name: "TF-IDF + Logistic Regression" })).toBeVisible();
  await expect(page.getByText("99.65%")).toBeVisible();
  await expect(page.getByText("Not installed", { exact: true })).toHaveCount(2);
});
