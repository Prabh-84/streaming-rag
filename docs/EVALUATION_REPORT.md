# Streaming Live RAG — Quantitative Evaluation Report

Documentation only. This report synthesizes evaluation work already performed in this
repository — it introduces no new measurements, no new code, and no new benchmark runs beyond
what is cited below. Every number in this document is traceable to a specific persisted file,
test, or source module named alongside it.

## 1. Evaluation Objective

This evaluation measures whether the implemented pipeline satisfies the behavioral requirements
Theme 4 sets for a streaming RAG system, using the project's own benchmark harness
(`benchmarks/harness.py`) and test suite as the instrument:

- **Early retrieval** — does the Retrieval Controller begin retrieval before an utterance is
  marked final, when the transcript justifies it (Gate G2)?
- **Multi-intent handling** — is a compound request correctly decomposed into independent,
  gold-matching sub-queries (Gate G3)?
- **Factual grounding / citation safety** — is every citation traceable to real evidence, with
  zero fabricated document/chunk IDs (Gate G4, fabrication half only — see §8)?
- **Session refinement** — does a late-arriving constraint retrieve only the delta, without
  clearing session state (Gate G5)?
- **Telemetry coverage** — is every retrieval turn fully traceable in the event log (Gate G6)?
- **Retrieval quality** — what precision/recall does the hybrid retrieval path actually achieve
  on the scenarios that carry gold-relevant chunk labels?
- **Edge-case robustness** — what genuine failure modes, fixed or still-open, does direct
  inspection and reproduction surface beyond the pass/fail gates?

This report does **not** claim measurements the current harness does not produce (e.g. it does
not claim a measured Citation Grounding Rate, a measured G1, or any real-provider benchmark-suite
result — see §3, §6, §8).

## 2. Evaluation Environment

| Aspect | Value | Source |
|---|---|---|
| Benchmark suite | `benchmarks/streaming_suite_v1` | `benchmarks/streaming_suite_v1/scenarios.json` |
| Scenarios | 10 (`BENCH-01`..`BENCH-10`) | `scenarios.json` |
| Corpus | `benchmark_v1` — 4 synthetic documents, 10 chunks | `benchmarks/streaming_suite_v1/corpus/benchmark_v1/` |
| Retrieval architecture | Real in-memory Qdrant (dense) + real BM25 (sparse), RRF fusion, dedup, cross-encoder rerank | `app/retrieval/`, `app/reranking/cross_encoder.py` |
| Reranker | `FakeReranker` (offline mode) — deterministic word-overlap scoring | `tests/fakes.py`, `benchmarks/harness.py::_run_scenario` |
| Offline/fake mode | `--use-fakes`: `FakeEmbedder`, `FakeReranker`, `FakeDecomposer`, harness-local `EchoingGenerator` | `benchmarks/harness.py` |
| Generator/decomposer behavior | Deterministic doubles — `EchoingGenerator` quotes the first evidence chunk verbatim with its real citation tag; `FakeDecomposer` returns a fixed sub-query list only for BENCH-02, `None` (triggering the real fallback path) otherwise | `benchmarks/harness.py` |
| Result persistence | `benchmarks/results/<run_id>.json`, gitignored | `.gitignore` (`benchmarks/results/*`) |
| Reproducibility | Fresh `tempfile.TemporaryDirectory()` + fresh in-memory Qdrant client per run; `--use-fakes` runs are fully deterministic | `benchmarks/harness.py::run_benchmark` |

**Fake vs. production components:** retrieval itself (dense Qdrant + sparse BM25 + fusion +
reranking mechanics) is **real** in every run cited in this report, offline or not. Only the
embedder, reranker, decomposer, and generator are swapped for deterministic doubles under
`--use-fakes`. No run cited in this report made a real Gemini, Anthropic, or Groq API call.

## 3. Baseline Benchmark Results

Source: `benchmarks/results/run_1790299972.json` (persisted offline run, `--use-fakes`).

| Metric | Value | Target | Result | Meaning |
|---|---|---|---|---|
| Scenarios passed | 10/10 | — | — | Every `BENCH-01..10` scenario's gold decision sequence/sub-query/citation/uncertainty/contradiction expectations were met |
| G2 — Early Retrieval Rate | 1.0 | ≥ 0.80 | Pass | Fraction of early-retrieval-eligible scenarios where `RETRIEVAL_STARTED` preceded the utterance's final chunk |
| G3 — Multi-Intent Accuracy | 1.0 | ≥ 0.70 | Pass | Whether BENCH-02's compound utterance produced ≥ the gold sub-intent count |
| G4 fabrication rate | 0.0 | = 0.0 | Pass | Fraction of `CITATION_CREATED` events whose `doc_id` is the injected non-existent id (`nonexistent_doc`) |
| G5 — Session Refinement Accuracy | 1.0 | verified state continuity | Pass | Whether late-constraint scenarios (BENCH-06, BENCH-09) retrieved only the delta, with no session-state clear |
| G6 — Telemetry Coverage | 1.0 | = 1.0 | Pass | Every scenario produced a non-empty trace, and every `RETRIEVE` decision shares its `trace_id` with ≥1 downstream event |

**G1 is not reported here.** `benchmarks/harness.py::_aggregate_gates` computes G2, G3,
G4_fabrication, G5, and G6 only — it does not compute or persist a G1 value in any result file
inspected for this report. No G1 number is claimed.

## 4. Retrieval Quality

Precision/recall are computed per-scenario by `benchmarks/harness.py::_grade_scenario` from real
`RETRIEVAL_COMPLETED` events against `scenarios.json`'s `gold_relevant_chunk_ids`, using
`app/telemetry/metrics.py::retrieval_precision`/`retrieval_recall`. Values below are the actual
persisted results from Ablation #1 (§5) — the only measurements in this repository comparing
hybrid and dense-only retrieval directly.

| Scenario | Hybrid precision | Dense-only precision | Hybrid recall | Dense-only recall |
|---|---|---|---|---|
| BENCH-01 | 0.111 | 0.100 | 1.0 | 1.0 |
| BENCH-02 | 0.250 | 0.200 | 1.0 | 1.0 |
| BENCH-06 | 0.143 | 0.100 | 1.0 | 1.0 |
| BENCH-07 | 0.125 | 0.100 | 1.0 | 1.0 |
| BENCH-08 | 0.000 | 0.000 | n/a (empty gold set) | n/a (empty gold set) |
| BENCH-09 | 0.286 | 0.200 | 1.0 | 1.0 |
| BENCH-03/04/05/10 | not reported — `scenarios.json` defines no `gold_relevant_chunk_ids` for these | | | |

No aggregate precision/recall metric (e.g. a mean across scenarios) is calculated here, because
`benchmarks/harness.py` does not define or report one.

## 5. Ablation #1 — Hybrid vs. Dense-only

**CONTROL:** Dense (Qdrant) + BM25 → RRF fusion → dedup → reranking (unmodified
`run_benchmark(use_fakes=True)`).

**ABLATION:** Dense-only → reranking, via `app.retrieval.sparse_bm25.SparseIndexRegistry.search`
monkeypatched to return `[]` for the run's duration only, restored afterward
(`scripts/run_ablation_hybrid_vs_dense.py`).

**Held fixed in both arms:** corpus (`benchmark_v1`), all 10 scenarios/transcripts/`send_order`,
reranker (`FakeReranker`), evidence limits (`Settings`, untouched), grounding validator (real,
unmodified), generator/decomposer (`EchoingGenerator`/`FakeDecomposer`, identical in both arms),
and the harness itself (`benchmarks/harness.py`, unmodified, called identically both times).

**Results** (`benchmarks/results/ablation_hybrid_vs_dense_only/{baseline_hybrid,ablation_dense_only}/run_*.json`):

| | Hybrid (baseline) | Dense-only (ablation) |
|---|---|---|
| Scenarios passed | 10/10 | 10/10 |
| G2 | 1.0 | 1.0 |
| G3 | 1.0 | 1.0 |
| G4 fabrication | 0.0 | 0.0 |
| G5 | 1.0 | 1.0 |
| G6 | 1.0 | 1.0 |

**Interpretation:** removing BM25 did not change any gate-level outcome or any scenario's
pass/fail status. Retrieval precision was equal to or lower under dense-only in every scored
scenario (see §4's table), while recall stayed at 1.0 in both arms for every scored scenario. No
overall "winner" is declared: this 10-scenario, 10-chunk benchmark is too small and too curated
to establish which configuration is superior at any realistic scale — it only demonstrates that,
on this specific suite, hybrid retrieval retrieves the same relevant chunks with measurably less
irrelevant noise, without this affecting the outcome gates.

## 6. Ablation #2 — Rule-based vs. Model-based Controller

**Not executed against the current implementation.**

Evidence gathered during inspection (full detail: prior audit turn, summarized here):

- The Retrieval Controller (`app/controller/retrieval_controller.py::on_chunk`) is the **only**
  implementation of the WAIT/RETRIEVE/NO_RETRIEVAL decision in this repository — deterministic,
  driven by embedding-stability cosine similarity and spaCy entity extraction, with zero LLM calls.
- `app/controller/suppression.py::_is_ambiguous` is a literal, documented placeholder that always
  returns `False`, with its own docstring stating LLM-based confirmation "is deferred to whichever
  later phase first wires in an LLM" — that phase does not exist in this codebase.
- No `ModelController`/`LLMController` class, function, or deterministic fake exists anywhere
  under `app/`, `tests/`, or `benchmarks/` (confirmed by repository-wide search).
- `benchmarks/harness.py`'s `_Harness` class exposes injection seams for exactly four components
  — `embedder`, `reranker`, `decomposer`, `generator` — and none for the controller itself.

Manufacturing a model-based controller solely to populate this ablation would mean authoring a
new, previously-nonexistent production decision path purely to generate a comparison — that would
not be a comparison of two existing implementations, it would be a comparison of the real
implementation against one invented for the occasion, with no independent basis for judging
whether the invented path is a fair or representative model-based baseline. This is not reported
as a benchmark failure: it is an accurate statement that this specific ablation's precondition
(two existing implementations to compare) is not met.

## 7. Edge-Case Failure Analysis

Full write-up: prior Edge-Case Failure Analysis turn. Summary, with each case's status classified
precisely:

### EC-01 — Single final chunk (FAILURE, FIXED)
A single `TranscriptChunk` with `is_final=true` and a real entity could never retrieve:
`compute_stability()` structurally returns `0.0` for fewer than `STABILITY_WINDOW` (default 2)
embeddings — numerically indistinguishable from genuine instability — so a session's first and
only chunk always read as "unstable" and WAITed forever. **This was a real, reproducible failure**
(confirmed via a real Docker+Groq end-to-end run before the fix). Fixed this session
(`app/controller/retrieval_controller.py`) by treating `chunk.is_final and insufficient_history`
as conclusive rather than unstable — scoped so it can only ever affect a session's very first
chunk, never real multi-chunk streaming. Verified by 4 new targeted tests
(`tests/unit/test_retrieval_controller.py`) plus a real end-to-end Groq re-verification.
**Current status: fixed, passing.**

### EC-02 — Compound-signal false positive (LIMITATION, offline-masked)
BENCH-01 and BENCH-10 (identical transcript, gold-labeled single-intent) both trigger
`has_compound_signal()`'s deterministic syntactic gate — directly observed as a
`decomposition_fallback`/`llm_returned_no_usable_sub_queries` telemetry event at the same
reproducible position in both the Ablation #1 baseline and dense-only run logs. The scenario still
passes only because the LLM-failure fallback (built for provider outages, not for this) happens to
produce the same single-sub-query result the gold label expects under `--use-fakes`'s
`FakeDecomposer(sub_queries=None)`. **The false-positive rate this gate would exhibit against a
real, well-behaved LLM provider is unmeasured** — a real decomposer might legitimately split this
text into 2 sub-queries, which would violate BENCH-01's own `max_sub_queries: 1` gold criterion.
**Current status: passing offline, real-provider behavior unverified.**

### EC-03 — Numeric-only contradiction detection (LIMITATION, by design)
BENCH-09 (two chunks disagreeing on a numeric venue capacity) passes:
`app/retrieval/fusion.py::flag_contradictions` correctly flags it via the corpus's declared
`venue` entity-key slot and `capacity` numeric slot. Its own docstring states this is
**"only meaningful for numeric slots... a corpus with no entity-key slot simply has no
contradiction detection."** A categorical/qualitative contradiction (e.g. two chunks disagreeing
on a yes/no policy fact) is structurally invisible to this mechanism — not a bug, a documented
scope boundary. **Current status: working as designed for numeric slots; no coverage for
categorical contradictions.**

### EC-05 — `MAX_WAIT_CHUNKS` end-to-end coverage gap (COVERAGE GAP)
BENCH-05 sends exactly 3 ambiguous chunks; `MAX_WAIT_CHUNKS` defaults to 6, so no scenario in
`scenarios.json` ever reaches the forced-retrieve-after-max-wait branch
(`RetrievalTrigger.FORCED_AFTER_MAX_WAIT`). This path **is** unit-tested
(`tests/unit/test_retrieval_controller.py::test_forced_retrieve_after_max_wait_chunks`) but has
**zero coverage through the real WS/HTTP benchmark replay** the other 10 scenarios exercise.
**Current status: unit-tested, not benchmark-tested end-to-end.**

### EC-04 — Benchmark grading has no precision/recall gate (EVALUATION METHODOLOGY LIMITATION)
`benchmarks/harness.py::_grade_scenario` computes and stores `retrieval_precision`/
`retrieval_recall` but never compares either against a threshold to decide `passed`. Scenarios
pass with measured precision as low as 0.10–0.29 (§4). **Current status: methodology gap in the
harness, not a defect in retrieval itself.**

## 8. Benchmark Methodology Limitations

1. **Corpus is tiny:** `benchmark_v1` is 4 documents / 10 chunks — far smaller than any real
   deployment corpus, and small enough that `K_DENSE=K_SPARSE=20` effectively returns most or all
   of the corpus regardless of relevance.
2. **Recall ceiling:** every scored scenario already achieves recall 1.0 in the baseline, leaving
   no room for Ablation #1 to reveal a recall regression even if one existed at larger scale.
3. **Fake mode:** `FakeEmbedder`, `FakeReranker`, `FakeDecomposer`, and the harness-local
   `EchoingGenerator` are deterministic offline doubles used specifically to isolate the variable
   under test (retrieval strategy) — they are not the real cross-encoder/LLM behavior.
4. **Citation Grounding Rate is not measured.** Only `fabricated_citation_rate` is aggregated by
   `benchmarks/harness.py::_aggregate_gates`. The full `citation_grounding_rate` formula exists in
   `app/telemetry/metrics.py` but is not wired into any benchmark report in this repository.
5. **Retrieval precision/recall are calculated but not gated** — see EC-04.
6. **Ablation #2 is unavailable** without introducing a new controller architecture that does not
   currently exist (§6).
7. **Real-provider benchmark scope is limited.** This repository's evidence includes single real
   end-to-end requests against Gemini and Groq (verified working, per prior session work), and one
   real-provider benchmark-suite *attempt* that hit Gemini's free-tier rate limit within the first
   scenario. **No broad real-provider benchmark-suite evaluation is claimed** — only the specific,
   limited real-provider verification actually performed.

## 9. Requirements / Gate Coverage

| Requirement/Gate | Evidence | Status |
|---|---|---|
| Early retrieval (G2) | `benchmarks/results/run_1790299972.json`; Ablation #1 both arms | Measured, 1.0, pass |
| Multi-intent (G3) | `benchmarks/results/run_1790299972.json`; Ablation #1 both arms | Measured, 1.0, pass |
| Grounding / citation fabrication (G4) | `benchmarks/results/run_1790299972.json`; `tests/integration/test_benchmark_harness.py::test_grade_bench10_*` | Measured (fabrication only), 0.0, pass |
| Session refinement (G5) | `benchmarks/results/run_1790299972.json`; BENCH-06/BENCH-09 | Measured, 1.0, pass |
| Telemetry coverage (G6) | `benchmarks/results/run_1790299972.json` | Measured, 1.0, pass |
| Retrieval quality (precision/recall) | Ablation #1 result JSONs, §4 | Measured per-scenario; no pass/fail gate exists (EC-04) |
| Edge-case analysis | §7 (EC-01..EC-05) | 1 failure found and fixed; 4 documented limitations/coverage gaps |
| Ablation #1 (hybrid vs. dense-only) | §5 | Executed, both arms complete, results compared |
| Ablation #2 (rule-based vs. model-based controller) | §6 | Not executed — no model-based controller exists in this repository |
| G1 (reproducibility) | Not computed by the harness in any inspected result file | Not reported |

## 10. Reproducibility

Commands actually run and verified during this evaluation work:

```bash
python scripts/run_benchmark.py --use-fakes
python scripts/run_ablation_hybrid_vs_dense.py
python -m pytest -q
python -m pytest tests/unit/test_hybrid.py tests/integration/test_benchmark_harness.py -q
```

Each `run_benchmark()` call uses a fresh `tempfile.TemporaryDirectory()` and a fresh in-memory
Qdrant client, so successive runs never share state; `--use-fakes` mode is fully deterministic
(no network, no LLM API key required). Results persist to `benchmarks/results/<run_id>.json`
(baseline) or the caller-supplied `results_dir` (Ablation #1's two arms) — all gitignored by
`benchmarks/results/*`.

## 11. Overall Findings

The current benchmark suite demonstrates successful behavior across all five measured gates
(G2–G6) on the offline, deterministic run. Hybrid retrieval showed a measurable retrieval
precision advantage over dense-only on every scenario that carries gold-relevant chunk labels,
while dense-only retained perfect recall on this small benchmark — so the ablation shows a real,
measured difference without either configuration failing any gate. One genuine controller failure
(EC-01) was discovered through direct inspection and reproduction, fixed, and verified with
targeted tests and a real end-to-end run; it is not an open issue in the current codebase. Several
architectural and evaluation-methodology limitations remain open and are documented rather than
hidden: a compound-signal false-positive whose real-provider rate is unmeasured (EC-02),
numeric-only contradiction detection (EC-03), no end-to-end coverage of the forced-retrieve path
(EC-05), and no precision/recall pass/fail gate in the harness itself (EC-04). The benchmark's
small, curated size (10 scenarios, 10 chunks) limits how far any of these results generalize to a
production-scale corpus. The rule-based-vs-model-based controller ablation was not executable
because no model-based controller implementation exists in this repository — this is reported as
a missing precondition, not a failed experiment.

## 12. Future Work

- A broader, more adversarial benchmark corpus, large enough to separate hybrid and dense-only
  retrieval at the gate level, not only at the precision-metric level.
- A retrieval precision/recall pass/fail gate in `benchmarks/harness.py`, closing EC-04.
- Categorical (not just numeric) contradiction detection in `app/retrieval/fusion.py`, closing
  the scope boundary documented in EC-03.
- An end-to-end benchmark scenario that actually reaches `MAX_WAIT_CHUNKS`, closing the EC-05
  coverage gap.
- A measurement of the compound-signal gate's real false-positive rate against a real LLM
  provider, to quantify EC-02's currently-unmeasured risk.
- A legitimate model-based controller implementation, deliberately added as a real architectural
  extension (not manufactured solely for an ablation), followed by a properly isolated Ablation #2
  once both implementations genuinely exist side by side.

None of the above has been implemented as part of this report.
