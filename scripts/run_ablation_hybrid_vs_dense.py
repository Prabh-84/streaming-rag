"""Ablation #1 driver: Hybrid (dense + BM25) vs. dense-only retrieval.

    python scripts/run_ablation_hybrid_vs_dense.py

This is an isolated experiment driver, not a permanent feature. It does not modify anything
under app/, does not add a config flag, and does not touch benchmarks/harness.py or
scenarios.json - it only calls the *existing* benchmarks.harness.run_benchmark() twice:

- BASELINE: unmodified - real dense (Qdrant) + real BM25 (sparse_bm25.py), fused (RRF),
  deduped, and reranked exactly as scripts/run_benchmark.py --use-fakes already does.
- ABLATION: app.retrieval.sparse_bm25.SparseIndexRegistry.search is monkeypatched, for the
  duration of this one run only, to return [] - the *same* "a search mode contributes no
  evidence" path app.retrieval.hybrid.HybridRetriever._run_mode already handles gracefully
  and is already covered by tests/unit/test_hybrid.py::test_missing_sparse_index_is_recorded_
  not_raised. This degrades retrieval to dense-only without touching production code. The
  original method is restored in a `finally` block, so it is restored even if the ablation
  run itself raises.

Both arms run with use_fakes=True (deterministic embedder/reranker/decomposer/generator - no
network, no LLM API key, no Gemini/Anthropic/Groq call of any kind). Real Qdrant (in-memory)
and real BM25 are used for retrieval in both arms, exactly as --use-fakes already does for the
existing benchmark suite - only sparse availability differs between the two runs.

Same corpus (benchmark_v1), same 10 scenarios.json scenarios, same transcripts/send_order, same
reranker/evidence limits/grounding validator/generator-decomposer configuration in both arms,
since both are just two calls to the one unmodified run_benchmark() entry point.

Results are written to two separate, already-gitignored directories under benchmarks/results/,
never overwriting the regular benchmark suite's own output.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # run as `python scripts/run_ablation_hybrid_vs_dense.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval.sparse_bm25 import SparseIndexRegistry  # noqa: E402
from benchmarks.harness import RESULTS_DIR, run_benchmark  # noqa: E402

ABLATION_ROOT = RESULTS_DIR / "ablation_hybrid_vs_dense_only"
BASELINE_DIR = ABLATION_ROOT / "baseline_hybrid"
ABLATION_DIR = ABLATION_ROOT / "ablation_dense_only"


async def _empty_sparse_search(
    self: SparseIndexRegistry,
    corpus_id: str,
    text: str,
    top_k: int,
    *,
    timeout_ms: int | None = None,
) -> list:
    """Drop-in replacement for SparseIndexRegistry.search: BM25 contributes no evidence, as if
    sparse retrieval were disabled, without touching the real index or any application code."""
    return []


def main() -> int:
    print("=== ABLATION #1: Hybrid (dense + BM25) vs. dense-only retrieval ===\n")

    print(">>> BASELINE: real dense + real BM25 (unmodified run_benchmark)")
    baseline_report = run_benchmark(use_fakes=True, results_dir=BASELINE_DIR)
    baseline_path = BASELINE_DIR / (baseline_report.run_id + ".json")
    print(f"    run_id={baseline_report.run_id}  -> {baseline_path}")

    print("\n>>> ABLATION: SparseIndexRegistry.search patched to return [] (dense-only)")
    original_search = SparseIndexRegistry.search
    SparseIndexRegistry.search = _empty_sparse_search
    try:
        ablation_report = run_benchmark(use_fakes=True, results_dir=ABLATION_DIR)
    finally:
        SparseIndexRegistry.search = original_search
    ablation_path = ABLATION_DIR / (ablation_report.run_id + ".json")
    print(f"    run_id={ablation_report.run_id}  -> {ablation_path}")

    restored_ok = SparseIndexRegistry.search is original_search
    print(f"\nSparseIndexRegistry.search restored to original: {restored_ok}")
    if not restored_ok:
        print("WARNING: sparse search was not restored correctly.", file=sys.stderr)
        return 1

    print("\nBaseline gates:", baseline_report.gates)
    print("Ablation gates:", ablation_report.gates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
