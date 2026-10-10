import { test, expect, type Page } from "@playwright/test";

const url = "http://127.0.0.1:8769";
const program = `import cutlass
import cutlass.cute as cute
import cuteviz

@cute.jit
def custom_entry(rows: cutlass.Constexpr):
    tile = cute.make_layout((rows, 4), stride=(4, 1))
    cuteviz.inspect("my_tile", tile)

def custom_args():
    return (3,)
`;

async function edit(page: Page, text: string) {
  const editor = page.getByRole("textbox", { name: "Python kernel editor" });
  await editor.click();
  await editor.press("ControlOrMeta+a");
  await page.keyboard.insertText(text);
}

test.describe("local Python workbench", () => {
  test.beforeEach(async ({ page, request }) => {
    const info = await (await request.get(`${url}/api/workbench`)).json();
    test.skip(
      !info.can_compile,
      "Requires cuteviz[capture] in the server environment",
    );
    await page.goto(url);
    await expect(
      page.getByRole("textbox", { name: "Python kernel editor" }),
    ).toBeVisible();
  });

  test("edit, compile, download, recompile, recover from errors, and persist a draft", async ({
    page,
  }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await edit(page, program);
    await page
      .getByLabel("Entry function", { exact: true })
      .fill("custom_entry");
    await page
      .getByLabel("Arguments factory", { exact: true })
      .fill("custom_args");
    await page.getByLabel("Target architecture", { exact: true }).fill("sm_80");
    await page.screenshot({
      path: "test-results/workbench-editor.png",
      fullPage: true,
    });
    await page
      .getByRole("textbox", { name: "Python kernel editor" })
      .press("ControlOrMeta+Enter");
    await expect(
      page.getByRole("heading", { name: "my_tile", exact: true }),
    ).toBeVisible({ timeout: 20000 });
    await expect(
      page.getByText("Draft differs from this capture.", { exact: false }),
    ).toHaveCount(0);
    await page
      .getByRole("gridcell", { name: "tensor coordinate 2,3, 11", exact: true })
      .click();
    await expect(page.getByLabel("Mapping explanation")).toContainText(
      "2 × 4 + 3 × 1 = 11",
    );
    await expect(page.locator(".source-active")).toContainText(
      'cuteviz.inspect("my_tile", tile)',
    );
    const downloadPromise = page.waitForEvent("download");
    await page
      .locator(".capture-toolbar")
      .getByRole("link", { name: "Download .cuteviz.json" })
      .click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/\.cuteviz\.json$/);
    const stream = await download.createReadStream();
    const chunks: Buffer[] = [];
    for await (const chunk of stream!) chunks.push(chunk);
    expect(JSON.parse(Buffer.concat(chunks).toString()).objects[0].label).toBe(
      "my_tile",
    );
    await page.getByRole("button", { name: "Back to editor" }).click();
    await edit(page, program.replace("(3,)", "(4,)"));
    await page.getByRole("tab", { name: /Inspect/ }).click();
    await expect(
      page.getByText("Draft differs from this capture.", { exact: false }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Compile & capture" }).click();
    await expect(
      page.getByRole("gridcell", {
        name: "tensor coordinate 3,3, 15",
        exact: true,
      }),
    ).toBeVisible({ timeout: 20000 });
    await page.screenshot({
      path: "test-results/workbench-inspection.png",
      fullPage: true,
    });
    await page.getByRole("button", { name: "Back to editor" }).click();
    await edit(page, "def invalid(:\n    pass\n");
    await page.getByRole("button", { name: "Compile & capture" }).click();
    await expect(page.getByLabel("Compiler output")).toContainText(
      "SyntaxError",
      { timeout: 20000 },
    );
    await page.getByRole("tab", { name: /Inspect/ }).click();
    await expect(
      page.getByRole("gridcell", {
        name: "tensor coordinate 3,3, 15",
        exact: true,
      }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Back to editor" }).click();
    await expect(page.getByText("Draft saved locally")).toBeVisible();
    await page.reload();
    await expect(
      page.getByRole("textbox", { name: "Python kernel editor" }),
    ).toContainText("def invalid(:");
    expect(errors).toEqual([]);
  });

  test("cancel a running process and open a Python file", async ({ page }) => {
    await page
      .locator('input[type="file"]')
      .setInputFiles({
        name: "slow.py",
        mimeType: "text/x-python",
        buffer: Buffer.from(
          'import time\nprint("waiting", flush=True)\ntime.sleep(60)\n',
        ),
      });
    await expect(
      page.getByRole("textbox", { name: "Python kernel editor" }),
    ).toContainText("time.sleep(60)");
    await page.getByRole("button", { name: "Compile & capture" }).click();
    await expect(
      page.getByRole("button", { name: "Cancel compilation" }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Cancel compilation" }).click();
    await expect(page.getByRole("status")).toHaveText("Cancelled", {
      timeout: 10000,
    });
    await expect(
      page.getByRole("button", { name: "Compile & capture" }),
    ).toBeEnabled();
  });
});
