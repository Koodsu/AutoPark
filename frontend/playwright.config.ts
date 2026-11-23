import { defineConfig } from "@playwright/test";
import { existsSync } from "node:fs";

// Use the pre-installed Chromium when present (no `playwright install` needed there).
const preinstalled = "/opt/pw-browsers/chromium";
const executablePath =
  process.env.PW_CHROMIUM ?? (existsSync(preinstalled) ? preinstalled : undefined);

export default defineConfig({
  testDir: "tests/e2e",
  timeout: 150_000,
  retries: 0,
  use: {
    baseURL: "http://localhost:5173",
    viewport: { width: 1500, height: 900 },
    launchOptions: executablePath ? { executablePath } : {},
  },
  webServer: [
    {
      command: "cd ../backend && uv run parkpilot serve --port 8000",
      url: "http://127.0.0.1:8000/api/health",
      reuseExistingServer: true,
      timeout: 120_000,
      env: { PARKPILOT_PREFS: "../frontend/test-results/prefs.json" },
    },
    {
      command: "pnpm dev",
      url: "http://localhost:5173",
      reuseExistingServer: true,
      timeout: 60_000,
    },
  ],
});
