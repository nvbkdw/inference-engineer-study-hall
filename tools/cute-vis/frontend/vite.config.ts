import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
  plugins: [svelte()],
  build: {
    outDir: "../src/cuteviz/static",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (
            id.includes("/codemirror/") ||
            id.includes("/@codemirror/") ||
            id.includes("/@lezer/")
          )
            return "python-editor";
        },
      },
    },
  },
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
        configure(proxy) {
          proxy.on("proxyReq", (request, original) => {
            // Same-origin browser requests arrive via the local Vite dev server.
            // Preserve foreign origins so the API rejects them.
            if (original.headers.origin === `http://${original.headers.host}`)
              request.setHeader("Origin", "http://127.0.0.1:8765");
          });
        },
      },
    },
  },
});
