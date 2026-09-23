"""TelemetryEvent model (PRD_TRD.md §7.10).

Re-exports the envelope defined in app.core.events rather than redefining it, so the event schema
has exactly one source of truth (docs/TELEMETRY.md §1) instead of two definitions that could drift.
"""

from __future__ import annotations

from app.core.events import EventType, TelemetryEvent

__all__ = ["EventType", "TelemetryEvent"]
