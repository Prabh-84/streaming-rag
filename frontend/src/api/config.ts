/**
 * Backend location + the demo API key. The key is never hardcoded and never bundled: it lives
 * only in sessionStorage, entered once by whoever is running the demo (see ConnectPanel). This
 * is the application's own bearer token (REQ-SEC-04) - not a provider credential - but it still
 * never appears in source or in the built JS bundle.
 */

const API_KEY_STORAGE_KEY = "streaming-rag.api-key";

export const apiBaseUrl: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8000";

export const wsBaseUrl: string =
  (import.meta.env.VITE_WS_BASE_URL as string | undefined) ?? "ws://localhost:8000";

export const defaultCorpusId: string =
  (import.meta.env.VITE_DEFAULT_CORPUS_ID as string | undefined) ?? "northstar_demo_extended";

export function getStoredApiKey(): string {
  try {
    return sessionStorage.getItem(API_KEY_STORAGE_KEY) ?? "";
  } catch {
    return "";
  }
}

export function setStoredApiKey(key: string): void {
  try {
    if (key) sessionStorage.setItem(API_KEY_STORAGE_KEY, key);
    else sessionStorage.removeItem(API_KEY_STORAGE_KEY);
  } catch {
    // sessionStorage unavailable (private mode, etc.) - the key just won't persist across reloads.
  }
}
