"""REQ-EVAL-01 (HC-2, No Hardcoding): the held-out benchmark suite must never leak into
application code. `scripts/run_benchmark.py` and `benchmarks/harness.py` are the only places
allowed to know the suite's name or fixture content (docs/EVALUATION.md §4) — they sit outside
`app/` specifically so this boundary is mechanically checkable.

This is a static safeguard, not the Phase 9 evaluation harness itself (not implemented yet).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "app"
BENCHMARKS_ROOT = REPO_ROOT / "benchmarks"

# Stable scenario identifiers named throughout the frozen spec (PRD_TRD.md §15.1 / original TRD
# Part 8, BENCH-01..BENCH-10). Project-specific literal tags with no legitimate reason to appear
# in application source — a match means something is branching on a specific benchmark case.
BENCH_CASE_IDS = tuple(f"BENCH-{i:02d}" for i in range(1, 11))

# The held-out test-set's own name (docs/EVALUATION.md §4). app/ must stay agnostic to which
# suite is running; only the scripts/ and benchmarks/ tooling may know it.
SUITE_NAME = "streaming_suite_v1"

FORBIDDEN_LITERALS = (*BENCH_CASE_IDS, SUITE_NAME)

# Fixture-value strings shorter than this are skipped when checking benchmark fixture content
# against app/ source, specifically to avoid false positives on generic technical vocabulary
# ("text", "chunk_id", "venue", "provisional", ...) that legitimately appears in both.
_MIN_FIXTURE_STRING_LEN = 20


def _app_source_files() -> list[Path]:
    return sorted(APP_ROOT.rglob("*.py"))


def test_no_benchmark_strings_in_app():
    """No file under app/ may reference a BENCH-nn scenario id or the held-out suite's name —
    the concrete form REQ-EVAL-01's "no branching on a specific benchmark test-case identifier"
    would take."""
    offenders = []
    for path in _app_source_files():
        text = path.read_text(encoding="utf-8")
        for literal in FORBIDDEN_LITERALS:
            if literal in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {literal!r}")
    assert not offenders, "benchmark-specific literal(s) found in app/:\n" + "\n".join(offenders)


def _fixture_files() -> list[Path]:
    """Gold-data files under benchmarks/, excluding tooling (harness.py) and run output
    (results/). Returns [] today since benchmarks/streaming_suite_v1/ doesn't exist until
    Phase 9 — that is expected, not a test bug."""
    if not BENCHMARKS_ROOT.is_dir():
        return []
    files = []
    for path in BENCHMARKS_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in (".json", ".jsonl", ".yaml", ".yml"):
            continue
        if "results" in path.relative_to(BENCHMARKS_ROOT).parts:
            continue
        files.append(path)
    return files


def _string_values(obj: Any) -> list[str]:
    """Recursively collect string leaf values from parsed JSON/YAML fixture content."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _string_values(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _string_values(v)]
    return []


def _load_fixture(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(text)
    return json.loads(text)


def test_no_benchmark_fixture_content_in_app_source():
    """Forward-compatible companion to the check above: once Phase 9 adds real gold fixtures
    under benchmarks/, none of their distinctive string values (gold queries, sub-intent labels,
    answers) may appear verbatim in app/ source. No fixtures exist yet, so this checks zero files
    today by design and activates automatically once benchmarks/streaming_suite_v1/ gains
    content — no edits needed here when that happens.
    """
    distinctive_strings: set[str] = set()
    for path in _fixture_files():
        for value in _string_values(_load_fixture(path)):
            if len(value) >= _MIN_FIXTURE_STRING_LEN:
                distinctive_strings.add(value)

    if not distinctive_strings:
        return  # nothing to check yet - benchmarks/streaming_suite_v1/ is still empty

    app_text = "\n".join(p.read_text(encoding="utf-8") for p in _app_source_files())
    offenders = [s for s in distinctive_strings if s in app_text]
    assert not offenders, f"benchmark fixture content embedded in app/ source: {offenders}"
