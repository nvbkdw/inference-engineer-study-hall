import { test, expect } from "@playwright/test";
import { spawn } from "node:child_process";

test("CTA tile footprints and coordinate selection link to the recovered global tensor", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const server = spawn("../.venv/bin/cuteviz", [
    "serve",
    "../examples/captures/tiles.cuteviz.json",
    "--port",
    "8771",
    "--read-only",
  ]);
  try {
    await expect
      .poll(async () => {
        try {
          return (await fetch("http://127.0.0.1:8771/api/capture")).status;
        } catch {
          return 0;
        }
      })
      .toBe(200);
    await page.goto("http://127.0.0.1:8771");
    const sidebar = page.getByRole("navigation", { name: "Captured objects" });
    await sidebar.getByRole("button", { name: /^T cta_tile tensor/ }).click();
    await expect(page.locator(".tensor-node")).toHaveCount(3);
    await page.getByLabel("Inspection blockIdx.x", { exact: true }).fill("1");
    await page
      .getByLabel("Inspection blockIdx.x", { exact: true })
      .press("Tab");
    await page.getByLabel("Inspection blockIdx.y", { exact: true }).fill("2");
    await page
      .getByLabel("Inspection blockIdx.y", { exact: true })
      .press("Tab");
    await page.getByRole("button", { name: "Fit all", exact: true }).click();
    const tile = page.getByRole("region", {
      name: "cta_tile · tensor",
      exact: true,
    });
    const backing = page.getByRole("region", {
      name: "cta_tile.backing · tensor",
      exact: true,
    });
    const local = tile.getByRole("gridcell", {
      name: "tensor coordinate 1,2, 66",
      exact: true,
    });
    await local.click();
    const global = backing.getByRole("gridcell", {
      name: "tensor coordinate 9,34, 610",
      exact: true,
    });
    await expect(global).toHaveAttribute("aria-selected", "true");
    await expect(backing.locator('[data-in-tile="true"]')).toHaveCount(128);
    await expect(backing.locator(".tile-footprint")).toHaveCount(1);
    await expect(page.getByLabel("Backing coordinate mapping")).toContainText(
      "(9, 34)",
    );
    await expect(page.getByLabel("Backing coordinate mapping")).toContainText(
      "2440",
    );
    await expect(
      page.getByRole("button", { name: "Links", exact: true }),
    ).toHaveAttribute("aria-pressed", "false");
    await backing
      .getByRole("gridcell", {
        name: "tensor coordinate 10,35, 675",
        exact: true,
      })
      .click();
    await expect(
      tile.getByRole("gridcell", {
        name: "tensor coordinate 2,3, 131",
        exact: true,
      }),
    ).toHaveAttribute("aria-selected", "true");
    await sidebar.getByRole("button", { name: /tile_column tensor/ }).click();
    await expect(page.locator(".tensor-node")).toHaveCount(3);
    await page.getByRole("button", { name: "Fit all", exact: true }).click();
    const column = page.getByRole("region", {
      name: "tile_column · tensor",
      exact: true,
    });
    await column
      .getByRole("gridcell", { name: "tensor coordinate 2, 128", exact: true })
      .click();
    const trail = page.getByLabel("Backing coordinate mapping");
    await expect(trail).toContainText("(2, 3)");
    await expect(trail).toContainText("(10, 35)");
    await trail.getByRole("button", { name: "Show in canvas" }).last().click();
    await expect(
      backing.getByRole("gridcell", {
        name: "tensor coordinate 10,35, 675",
        exact: true,
      }),
    ).toHaveAttribute("aria-selected", "true");
    await page.getByLabel("Inspection blockIdx.x", { exact: true }).fill("2");
    await page
      .getByLabel("Inspection blockIdx.x", { exact: true })
      .press("Tab");
    // The local slice is off-screen after navigation; its explanation still updates.
    await expect(trail).toContainText("(18, 35)");
    await page.getByRole("button", { name: "Fit all", exact: true }).click();
    await expect(backing.locator('[data-in-tile="true"]')).toHaveCount(8);
    await page.screenshot({
      path: "test-results/backing-tensor.png",
      fullPage: true,
    });
    await page
      .getByRole("button", { name: "Back to editor", exact: true })
      .click();
    await page.getByLabel("Example", { exact: true }).selectOption("tiles");
    await page
      .getByRole("button", { name: "Load example", exact: true })
      .click();
    await expect(page.locator(".cm-content")).toContainText(
      'cuteviz.inspect("cta_tile", tile)',
    );
    expect(errors).toEqual([]);
  } finally {
    server.kill();
  }
});
