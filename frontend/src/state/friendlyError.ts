import type { ErrorPayload } from "../api/types";

/** The backend's ERROR payload.message is the raw provider/exception string (useful for
 * developers, never for a judge on screen). This maps stage/error_type to a short, honest,
 * human-readable sentence for the UI - the raw message still goes to the console. */
export function friendlyErrorMessage(error: ErrorPayload): string {
  switch (error.stage) {
    case "decomposition":
      return "Could not split this into separate questions right now - continuing with a single combined query.";
    case "retrieval":
      return "A retrieval call failed and was skipped.";
    case "generation":
      return "The answer could not be generated right now (the language model provider is unavailable or rate-limited).";
    case "controller":
    case "stream":
      return "A processing step hit a temporary error and was skipped.";
    default:
      return "A temporary processing error occurred.";
  }
}
