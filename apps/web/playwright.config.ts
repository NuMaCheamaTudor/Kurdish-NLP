import { defineConfig, devices } from "@playwright/test";

// End-to-end smoke tests against the real API and real models.
// Starts both servers unless they are already running.
//   KNLP_PYTHON  Python with the API installed (default: ../../.venv/bin/python)
//   PW_CHANNEL   Use an installed browser, e.g. "chrome", instead of Playwright's Chromium

const python = process.env.KNLP_PYTHON ?? "../../.venv/bin/python";
const channel = process.env.PW_CHANNEL;

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  fullyParallel: false,
  reporter: [["list"]],
  use: {
    baseURL: "http://127.0.0.1:5173",
    trace: "retain-on-failure",
    ...(channel ? { channel } : {}),
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 }, ...(channel ? { channel } : {}) }, grepInvert: /@mobile/ },
    { name: "mobile", use: { ...devices["Pixel 7"], viewport: { width: 390, height: 844 }, ...(channel ? { channel } : {}) }, grep: /@mobile/ },
  ],
  webServer: [
    {
      command: `${python} -m uvicorn app.main:app --app-dir ../api --host 127.0.0.1 --port 8000`,
      url: "http://127.0.0.1:8000/health",
      reuseExistingServer: true,
      timeout: 60_000,
    },
    {
      command: "npx vite --host 127.0.0.1",
      url: "http://127.0.0.1:5173",
      reuseExistingServer: true,
      timeout: 60_000,
    },
  ],
});
