import { test, expect } from "@playwright/test";
import { readFileSync, mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawn } from "node:child_process";

test("canvas zoom, optional cross-tensor links, ownership and dragging", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(
    page.getByRole("region", { name: "Tensor canvas", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Fit all", exact: true }).click();
  await expect(page.locator(".tensor-node")).toHaveCount(2);
  await page
    .getByRole("navigation", { name: "Captured objects" })
    .getByRole("button", { name: /load_A copy/ })
    .click();
  await expect(page.locator(".tensor-node")).toHaveCount(4);
  await expect(
    page.getByRole("region", { name: "gA · tensor", exact: true }),
  ).toHaveCount(0);
  const zoom = page.getByLabel("Canvas zoom");
  const originalZoom = parseInt((await zoom.textContent())!);
  await page.getByRole("button", { name: "Zoom in", exact: true }).click();
  await expect
    .poll(async () => parseInt((await zoom.textContent())!))
    .toBeGreaterThan(originalZoom);
  await page.getByRole("button", { name: "Zoom out", exact: true }).click();
  await page.getByRole("button", { name: "Fit all", exact: true }).click();
  const source = page.getByRole("region", {
    name: "load_A.S · tensor",
    exact: true,
  });
  const target = page.getByRole("region", {
    name: "load_A.D · tensor",
    exact: true,
  });
  await source
    .getByRole("gridcell", { name: "tensor coordinate 1,2, 10", exact: true })
    .click();
  await expect(
    target.getByRole("gridcell", {
      name: "tensor coordinate 1,2, 10",
      exact: true,
    }),
  ).toHaveAttribute("aria-selected", "true");
  await expect(page.locator(".tensor-node")).toHaveCount(4);
  await expect(
    page
      .getByRole("navigation", { name: "Captured objects" })
      .getByRole("button", { name: /load_A copy/ }),
  ).toHaveClass(/active/);
  await expect(
    page
      .getByRole("region", { name: "gA · tensor", exact: true })
      .locator('[role="gridcell"][aria-selected="true"]'),
  ).toHaveCount(0);
  const links = page.getByRole("button", { name: "Links", exact: true });
  const redEdges = page.locator('.svelte-flow__edge[data-id^="selected:"]');
  await expect(links).toHaveAttribute("aria-pressed", "false");
  await expect(redEdges).toHaveCount(0);
  await links.click();
  await expect(redEdges).not.toHaveCount(0);
  await links.click();
  await expect(redEdges).toHaveCount(0);
  await expect(target.locator('[aria-selected="true"]')).not.toHaveCount(0);
  await page
    .getByRole("navigation", { name: "Captured objects" })
    .getByRole("button", { name: /load_A copy/ })
    .click();
  await page
    .getByRole("button", { name: "Select thread 0 value 0", exact: true })
    .click();
  await expect(
    page
      .getByRole("region", { name: "load_A · S", exact: true })
      .locator('[aria-selected="true"]'),
  ).not.toHaveCount(0);
  await expect(
    page
      .getByRole("region", { name: "load_A · D", exact: true })
      .locator('[aria-selected="true"]'),
  ).not.toHaveCount(0);
  await page.getByRole("button", { name: "Fit all", exact: true }).click();
  await expect(target.locator('[aria-selected="true"]')).not.toHaveCount(0);
  const heading = source.getByRole("button", {
    name: "Focus load_A.S tensor; drag to move",
  });
  const before = (await heading.boundingBox())!;
  await page.mouse.move(
    before.x + before.width / 2,
    before.y + before.height / 2,
  );
  await page.mouse.down();
  await page.mouse.move(
    before.x + before.width / 2 + 55,
    before.y + before.height / 2 + 35,
    { steps: 8 },
  );
  await page.mouse.up();
  await expect
    .poll(async () => (await heading.boundingBox())!.x)
    .toBeGreaterThan(before.x + 35);
  await page
    .getByRole("button", { name: "Expand canvas", exact: true })
    .click();
  await expect(page.locator(".canvas-expanded")).toBeVisible();
  await expect(
    page.locator(".grid-panel").filter({ hasText: "Updating…" }),
  ).toHaveCount(0);
  await page.screenshot({
    path: "test-results/tensor-canvas.png",
    fullPage: true,
  });
  await page.keyboard.press("Escape");
  await expect(page.locator(".canvas-expanded")).toHaveCount(0);
  expect(errors).toEqual([]);
});

test("sidebar selection limits the canvas to the selected object's relationships", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  const sidebar = page.getByRole("navigation", { name: "Captured objects" });
  const cases = [
    { name: /nested layout/, labels: ["nested · tensor"] },
    { name: /^T sA tensor/, labels: ["sA · tensor", "swizzled_sA · tensor"] },
    {
      name: /load_A.S tensor/,
      labels: [
        "load_A.S · tensor",
        "load_A · S",
        "load_A · D",
        "load_A.D · tensor",
      ],
    },
    {
      name: /bottom_left tensor/,
      labels: ["bottom_left · tensor", "gA · tensor"],
    },
  ];
  for (const item of cases) {
    await sidebar.getByRole("button", { name: item.name }).click();
    await page.getByRole("button", { name: "Fit all", exact: true }).click();
    await expect(page.locator(".tensor-node")).toHaveCount(item.labels.length);
    await expect
      .poll(async () =>
        (
          await page
            .locator(".tensor-node")
            .evaluateAll((nodes) =>
              nodes.map((node) => node.getAttribute("aria-label")),
            )
        ).sort(),
      )
      .toEqual([...item.labels].sort());
    await expect(sidebar.getByRole("button", { name: item.name })).toHaveClass(
      /active/,
    );
  }
  expect(errors).toEqual([]);
});

test("full tensor cards grow with their shape and resize without internal scrolling", async ({
  page,
}) => {
  await page.goto("/");
  await page
    .getByRole("button", { name: "Expand canvas", exact: true })
    .click();
  const panel = page.getByRole("region", { name: "gA · tensor", exact: true });
  const handle = panel.getByLabel("Resize gA tensor", { exact: true });
  const size = () =>
    panel.evaluate((element) => ({
      width: element.clientWidth,
      height: element.clientHeight,
    }));
  const before = await size();
  const shape = panel.getByRole("grid");
  await expect(shape).toHaveAttribute("aria-rowcount", "8");
  await expect(shape).toHaveAttribute("aria-colcount", "8");
  async function resize(dx: number, dy: number) {
    const box = (await handle.boundingBox())!;
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(
      box.x + box.width / 2 + dx,
      box.y + box.height / 2 + dy,
      { steps: 10 },
    );
    await page.mouse.up();
  }
  await resize(70, 40);
  const enlarged = await size();
  expect(enlarged.width).toBeGreaterThan(before.width + 60);
  expect(enlarged.height).toBeGreaterThan(before.height + 30);
  await resize(-40, -25);
  const smaller = await size();
  expect(smaller.width).toBeLessThan(enlarged.width - 30);
  expect(smaller.height).toBeLessThan(enlarged.height - 20);
  await expect(panel.getByRole("gridcell")).toHaveCount(64);
  const last = panel.getByRole("gridcell", {
    name: "tensor coordinate 7,7, 63",
    exact: true,
  });
  await last.click();
  await expect(last).toHaveAttribute("aria-selected", "true");
  expect(
    await panel.evaluate(
      (element) =>
        [...element.querySelectorAll("*")].filter((child) => {
          const style = getComputedStyle(child);
          return (
            ((style.overflowY === "auto" || style.overflowY === "scroll") &&
              child.scrollHeight > child.clientHeight + 1) ||
            ((style.overflowX === "auto" || style.overflowX === "scroll") &&
              child.scrollWidth > child.clientWidth + 1)
          );
        }).length,
    ),
  ).toBe(0);
  const saved = await size();
  await page.getByRole("button", { name: "Arrange", exact: true }).click();
  expect(await size()).toEqual(saved);
});

for (const [rows, columns] of [
  [128, 96],
  [1_000_000, 1_000_000],
]) {
  test(`full ${rows} × ${columns} tensor uses the canvas and bounded mapping requests`, async ({
    page,
  }) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    const directory = mkdtempSync(join(tmpdir(), "cuteviz-large-canvas-"));
    const filename = join(directory, "large.cuteviz.json");
    const capture = JSON.parse(
      readFileSync("../examples/captures/copy.cuteviz.json", "utf8"),
    );
    capture.objects = [capture.objects[0]];
    const tensor = capture.objects[0];
    tensor.label = "large_tensor";
    tensor.views[0].shape = [rows, columns];
    tensor.views[0].layout.shape = [rows, columns];
    tensor.views[0].layout.stride = [columns, 1];
    writeFileSync(filename, JSON.stringify(capture));
    const server = spawn("../.venv/bin/cuteviz", [
      "serve",
      filename,
      "--port",
      "8770",
      "--read-only",
    ]);
    try {
      await expect
        .poll(async () => {
          try {
            return (await fetch("http://127.0.0.1:8770/api/capture")).status;
          } catch {
            return 0;
          }
        })
        .toBe(200);
      const extents: number[][] = [];
      page.on("request", (request) => {
        if (request.url().includes("/api/mapping/query"))
          extents.push(request.postDataJSON().extent);
      });
      await page.goto("http://127.0.0.1:8770");
      const panel = page.getByRole("region", {
        name: "large_tensor · tensor",
        exact: true,
      });
      await expect(panel.getByRole("grid")).toHaveAttribute(
        "aria-rowcount",
        String(rows),
      );
      await expect(panel.getByRole("grid")).toHaveAttribute(
        "aria-colcount",
        String(columns),
      );
      expect(
        await panel.evaluate((element) => element.clientWidth),
      ).toBeGreaterThan(7000);
      expect(
        await panel.evaluate((element) => element.clientHeight),
      ).toBeGreaterThan(5000);
      if (rows === 128) {
        // More than one API page, all cells in the full tensor card, no manual paging.
        await expect(panel.getByRole("gridcell")).toHaveCount(rows * columns);
      } else {
        await expect(panel.locator(".tensor-overview")).toBeVisible();
        await expect(panel.getByRole("gridcell")).toHaveCount(0);
      }
      await page
        .getByLabel("Go to row", { exact: true })
        .fill(String(rows - 1));
      await page
        .getByLabel("Go to column", { exact: true })
        .fill(String(columns - 1));
      await page
        .getByRole("button", { name: "Go to cell", exact: true })
        .click();
      const last = panel.getByRole("gridcell", {
        name: `tensor coordinate ${rows - 1},${columns - 1}, ${rows * columns - 1}`,
        exact: true,
      });
      if (rows > 100_000) {
        // Exercise the same real pointer action as a user. At extreme scales
        // the canvas picker handles browsers whose SVG hit testing clamps.
        await expect(last).toBeVisible();
        const box = (await last.boundingBox())!;
        await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
      } else await last.click();
      await expect(last).toHaveAttribute("aria-selected", "true");
      await expect(page.getByLabel("Mapping explanation")).toContainText(
        String(rows * columns - 1),
      );
      await last.press("ArrowLeft");
      await expect(
        panel.getByRole("gridcell", {
          name: `tensor coordinate ${rows - 1},${columns - 2}, ${rows * columns - 2}`,
          exact: true,
        }),
      ).toHaveAttribute("aria-selected", "true");
      await page.getByRole("button", { name: "Fit all", exact: true }).click();
      await page
        .getByRole("button", { name: "Go to cell", exact: true })
        .click();
      await expect(last).toBeVisible();
      expect(extents.length).toBeGreaterThan(1);
      expect(extents.every((extent) => extent[0] * extent[1] <= 4096)).toBe(
        true,
      );
      await expect(panel.locator(".svg-scroll")).toHaveCount(0);
      expect(errors).toEqual([]);
    } finally {
      server.kill();
      await new Promise<void>((resolve) =>
        server.once("exit", () => resolve()),
      );
      rmSync(directory, { recursive: true, force: true });
    }
  });
}
