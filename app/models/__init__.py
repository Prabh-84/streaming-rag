"""Pydantic data models (PRD_TRD.md §7). One module per model, field lists frozen there.

`new_id()` is the shared ULID generator used by every PK field below — IDs are ULIDs unless
otherwise noted (PRD_TRD.md §7 preamble).
"""

from ulid import ULID


def new_id() -> str:
    return str(ULID())
