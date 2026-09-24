"""CLI entry point for the Phase 9 benchmark suite (docs/EVALUATION.md §4, Definition of Done).

    python scripts/run_benchmark.py              # real providers (whatever LLM_PROVIDER is set to)
    python scripts/run_benchmark.py --use-fakes  # deterministic, offline (CI / smoke-test mode)

Prints each scenario's pass/fail, the G2-G6 gate results, and the ten TELEMETRY.md §5 metrics
where computable, then writes the full report to benchmarks/results/<run_id>.json (gitignored -
run output, not suite data).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # run as `python scripts/run_benchmark.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.harness import run_benchmark  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Phase 9 held-out benchmark suite.")
    parser.add_argument(
        "--use-fakes",
        action="store_true",
        help="use deterministic offline test doubles instead of the real configured providers",
    )
    args = parser.parse_args(argv)

    report = run_benchmark(use_fakes=args.use_fakes)

    print(f"\nRun {report.run_id} against corpus_id={report.corpus_id!r}\n")
    for result in report.scenario_results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.scenario_id}: {result.description}")
        for failure in result.failures:
            print(f"         - {failure}")

    print("\nGates:")
    print(json.dumps(report.gates, indent=2))
    print("\nMetrics:")
    print(json.dumps(report.metrics, indent=2))

    all_scenarios_passed = all(r.passed for r in report.scenario_results)
    all_gates_passed = all(
        g.get("pass", True) for g in report.gates.values() if isinstance(g, dict)
    )
    return 0 if (all_scenarios_passed and all_gates_passed) else 1


if __name__ == "__main__":
    sys.exit(main())
