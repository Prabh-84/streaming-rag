import { apiBaseUrl } from "./config";
import type { CreateSessionRequest, CreateSessionResponse, SessionSnapshot } from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number | null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, apiKey: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${apiBaseUrl}${path}`, {
      ...init,
      headers: {
        Authorization: `Bearer ${apiKey}`,
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...init?.headers,
      },
    });
  } catch {
    throw new ApiError("Cannot reach the backend. Is it running?", null);
  }

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // response body wasn't JSON - keep the generic statusText
    }
    throw new ApiError(detail, response.status);
  }

  return (await response.json()) as T;
}

/** POST /session (docs/API.md §2). */
export function createSession(
  apiKey: string,
  payload: CreateSessionRequest,
): Promise<CreateSessionResponse> {
  return request<CreateSessionResponse>("/session", apiKey, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/** GET /session/{id} (docs/API.md §5). */
export function getSession(apiKey: string, sessionId: string): Promise<SessionSnapshot> {
  return request<SessionSnapshot>(`/session/${sessionId}`, apiKey);
}
