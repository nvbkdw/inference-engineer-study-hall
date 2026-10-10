import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";

test("source, coordinate calculation, copy linking, keyboard, and bank analysis", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "gA", exact: true }),
  ).toBeVisible();
  const cell = page
    .getByRole("region", { name: "gA · tensor", exact: true })
    .getByRole("gridcell", {
      name: "tensor coordinate 1,2, 10",
      exact: true,
    });
  await cell.click();
  await expect(page.getByLabel("Mapping explanation")).toContainText(
    "1 × 8 + 2 × 1 = 10",
  );
  await expect(page.getByLabel("Mapping explanation")).toContainText(
    "40 bytes",
  );
  await cell.press("ArrowRight");
  await expect(page.getByLabel("Mapping explanation")).toContainText(
    "1 × 8 + 3 × 1 = 11",
  );
  const activeSource = page.locator(".source-active");
  await expect(activeSource).toContainText('cuteviz.inspect("gA", gA)');
  await page
    .getByRole("navigation", { name: "Captured objects" })
    .getByRole("button", { name: /load_A copy/ })
    .click();
  await page.getByRole("gridcell", { name: /^S coordinate 0,0,/ }).click();
  await expect(
    page.getByRole("gridcell", { name: /^D coordinate 0,0,/ }),
  ).toHaveAttribute("aria-selected", "true");
  await expect(page.getByLabel("Mapping explanation")).toContainText(
    "→ D (0, 0)",
  );
  await page.getByLabel("Select thread", { exact: true }).fill("1");
  await expect(page.locator('[aria-selected="true"]')).not.toHaveCount(0);
  await page.getByText("Shared-memory access lab").click();
  await page.getByLabel("Lane byte offsets").fill("0, 128, 256, -");
  await page.getByRole("button", { name: "Analyze declared access" }).click();
  await expect(page.locator(".bank-result")).toContainText(
    "3 serialization round(s)",
  );
  await page.screenshot({
    path: "test-results/copy-inspection.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test("nested modes, slices, swizzle and source navigation", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByLabel("Search captured objects").fill("nested");
  await page
    .getByRole("navigation")
    .getByRole("button", { name: /nested layout/ })
    .click();
  const nested = page.getByRole("region", {
    name: "nested · tensor",
    exact: true,
  });
  await nested.getByLabel("tensor column axis").selectOption("2");
  await nested.getByLabel("tensor fix 0.1").fill("2");
  await expect(
    nested.getByRole("gridcell", {
      name: "tensor coordinate 1,2,3, 51",
      exact: true,
    }),
  ).toBeVisible();
  await page.getByLabel("Search captured objects").fill("swizzled");
  await page
    .getByRole("navigation")
    .getByRole("button", { name: /swizzled_sA tensor/ })
    .click();
  const swizzled = page.getByRole("region", {
    name: "swizzled_sA · tensor",
    exact: true,
  });
  await swizzled.getByRole("checkbox", { name: "Unswizzled offsets" }).check();
  await expect(
    swizzled.getByRole("gridcell", {
      name: "tensor coordinate 1,0, 8",
      exact: true,
    }),
  ).toBeVisible();
  await swizzled.getByLabel("tensor color").selectOption("bank");
  await expect(swizzled.locator(".legend")).toContainText("placement only");
  await swizzled
    .getByRole("checkbox", { name: "Unswizzled offsets" })
    .uncheck();
});

test("Hopper and Blackwell operands load and switch without a compiler", async ({
  page,
}) => {
  for (const arch of ["sm_90a", "sm_100a"]) {
    const capture = JSON.parse(
      readFileSync(`../examples/captures/${arch}.cuteviz.json`, "utf-8"),
    );
    const mma = capture.objects.find((o: any) => o.kind === "mma");
    // Metadata and mapping requests use a server holding the same portable capture.
    // Launch a separate server per target via an alternate port in the test runner.
    const { spawn } = await import("node:child_process");
    const port = arch === "sm_90a" ? 8767 : 8768;
    const server = spawn("../.venv/bin/cuteviz", [
      "serve",
      `../examples/captures/${arch}.cuteviz.json`,
      "--port",
      String(port),
    ]);
    try {
      await expect
        .poll(async () => {
          try {
            return (await fetch(`http://127.0.0.1:${port}/api/capture`)).status;
          } catch {
            return 0;
          }
        })
        .toBe(200);
      await page.goto(`http://127.0.0.1:${port}`);
      await page
        .getByRole("navigation")
        .getByRole("button", { name: /mma mma/ })
        .click();
      await expect(page.getByLabel("Operand A", { exact: true })).toBeVisible();
      await expect(page.getByLabel("Operand B", { exact: true })).toBeVisible();
      await page.getByRole("gridcell", { name: /^C coordinate 1,1,/ }).click();
      await expect(
        page.getByRole("gridcell", { name: /^A coordinate 1,1,/ }),
      ).toHaveAttribute("aria-selected", "true");
      await page.getByLabel("Active operand").selectOption("B");
      if (arch === "sm_100a")
        await expect(page.getByLabel("Mapping explanation")).toContainText(
          "tensor-memory",
        );
      else
        await expect(page.getByLabel("Mapping explanation")).toContainText(
          "thread",
        );
      await page.getByLabel("Go to row").fill("8");
      await page.getByLabel("Go to column").fill("0");
      await page
        .getByRole("button", { name: "Go to cell", exact: true })
        .click();
      await expect(
        page.getByRole("gridcell", { name: /^B coordinate 8,0,/ }),
      ).toBeVisible();
      expect(mma.views).toHaveLength(3);
    } finally {
      server.kill();
    }
  }
});
