import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server binds 0.0.0.0 so it is reachable from outside a Docker container; the backend URL
// itself is never hardcoded here - see src/api/config.ts, which reads it from Vite env vars.
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
  },
  preview: {
    host: true,
    port: 5173,
  },
});
