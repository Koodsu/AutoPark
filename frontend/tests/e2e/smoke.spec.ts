import { expect, test } from "@playwright/test";

test("loads a scenario and the car reaches PARKED", async ({ page }) => {
  await page.goto("/?scenario=charger_stall");
  const start = page.getByTestId("start");
  await expect(start).toBeEnabled();
  await expect(page.getByTestId("scenario")).toHaveValue("charger_stall");
  await start.click();
  const banner = page.getByTestId("state-banner");
  await expect(banner).toHaveAttribute("data-state", /.+/, { timeout: 15_000 });
  await page.getByTestId("speed").selectOption("4");
  // The ranked spot list and the canvas are live while it drives.
  await expect(page.getByTestId("spot-list")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId("world-canvas")).toBeVisible();
  await expect(banner).toHaveAttribute("data-state", "PARKED", { timeout: 120_000 });
  await expect(banner).toContainText("charge port");
});
