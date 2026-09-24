import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  use: {
    baseURL: "http://127.0.0.1:8766",
    viewport: { width: 1440, height: 1000 },
  },
  webServer: [
    {
      command:
        "../.venv/bin/cuteviz serve ../examples/captures/copy.cuteviz.json --port 8766",
      url: "http://127.0.0.1:8766",
      reuseExistingServer: !process.env.CI,
    },
    {
      command:
        "../.venv/bin/cuteviz serve --port 8769 --output-dir test-results/runs",
      url: "http://127.0.0.1:8769",
      reuseExistingServer: !process.env.CI,
    },
  ],
});
