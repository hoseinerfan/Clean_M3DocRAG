# Controlled graph-to-answer timing: scope and execution

Prepared 2026-09-18. Original paper/Overleaf and saved predictions are unchanged.
This experiment is **not full online query-to-answer runtime** and does not
by itself finish reviewer R1.2. It is the controlled downstream comparison
that can now be run using the replayed graph configurations and existing reader.

## Completed precheck

The author supplied successful accounting/stdout for graph replay job 15911730:
`COMPLETED|0:0|00:00:31|node011`. Main Exact-MaxSim/no-hyperlink GPP and each
of the three legacy auxiliary branches matched all 16 sampled questions:
candidate sets, complete 1000-page orders, and scores (rtol 1e-6, atol 1e-8).
These are 16 distinct questions checked in four branches, not 64 questions.
HPC report: `output/racs_graph_replay_15911730/replay.json`.
Its full JSON has not been transferred locally; the next job reads it directly
and verifies the audit/input fingerprints before generating new reports.

## What is measured

| Stage | GPP condition | Full-CAPP condition |
| --- | --- | --- |
| Upstream dense and SPLADE rankings | Cached, outside timer | Cached, including distinct legacy dense input, outside timer |
| Main no-hyperlink graph | Recomputed inside timer | Recomputed inside timer |
| Three legacy auxiliary graphs | Not needed | Recomputed inside timer |
| Source-map preparation and CAPP features/scoring/blending | Not needed | Inside timer; saved 30-feature model, alpha 0.40 |
| PDF rendering/image selection | Inside timer | Inside timer |
| Four-page Qwen reader, including prompt/image processing | Inside timer | Inside timer |
| Correctness checks and serialization | Outside timer | Outside timer |

No training occurs. The job writes only a new job-specific directory.
It does not regenerate dense or sparse retrieval. Required upstream work is
not free: its exclusion means neither the absolute times nor their difference
can be called the full historical system's online latency/overhead.

## Controls and interpretation

- 128 measured questions selected by fixed SHA256 order, independent of results;
  four additional, disjoint questions for warm-up in every worker.
- Same single allocated GPU, same qids and same question order. Four repetitions
  per method, executed GPP/CAPP, CAPP/GPP, GPP/CAPP, CAPP/GPP. Each of these eight
  workers is a fresh process. Workers run sequentially, never concurrently.
- 16-bit Qwen2-VL-7B-Instruct, same existing reader code/prompt, greedy decoding,
  at most 128 generated tokens, exactly four unique pages. No model download.
- One CPU math thread; allocation reserves eight CPUs and 64 GB. Record actual
  CPU/GPU/driver, library versions, reader dtype/attention and configuration hashes.
  Checkpoint weight files are not byte-hashed; this is stated in the report.
- Explicit GPU synchronization before and after each timed query. Stage timings
  decompose graph work, source/CAPP work, image preparation and reader generation.
- No persistent rendered-document cache across questions; render a document only
  once within a question if multiple selected pages share it. This is deliberately
  different from the historical reader's 16-document LRU cache. OS filesystem cache
  and other node load remain uncontrolled; balanced order reduces, not eliminates,
  order effects. There is no claimed cold-cache or service-throughput measurement.
- Every graph's full candidate order and scores must match, and each full CAPP
  order must match, including warm-up questions. Checks run outside the timer.
  A mismatch fails the experiment rather than selecting another favorable sample.
- Mean/median/p95 are calculated over per-question averages across four repeats.
  Per-pass summaries and per-query measurements are retained. These p95 values
  are not the p95 of all individual requests or a concurrent service's latency.
- CPU peak RSS includes initialization, inputs and small validation references.
  GPU peak allocated/reserved values are reset after warm-up, but include resident
  weights and retained allocator cache. Report each fresh worker, not differences
  between high-water counters from one reused process. Sampled upstream bundles
  mean these are benchmark memory figures, not full-corpus deployment requirements.
- Separate rank-input and reader/dataset initialization durations are recorded;
  they exclude Python imports and bundle parsing and are not full cold-start times.
  Parent preparation is reported separately and excluded from condition RSS.
- Answer variation across repeated generations is counted. Do not replace the
  paper's full-cohort EM/F1 with the timing subset's generated answers.

## Submit on HPC after pulling this commit

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
if [ "$(git branch --show-current)" = "codex/mmdocir-hpc-workflow" ]; then
  git pull --ff-only origin codex/mmdocir-hpc-workflow &&
  mkdir -p output &&
  sbatch examples/sbatch_racs_graph_reader_runtime.sh
else
  echo "Stop: unexpected branch."
  git branch --show-current
fi
```

One GPU job, six-hour cap. Do not resubmit merely because it is pending.
Save its job ID. After completion, inspect accounting and the last log lines.
The report is `output/racs_graph_reader_runtime_JOBID/runtime.json`, with
status `validated_cached_retrieval_graph_to_answer`. On failure, inspect
`failure.json` and stderr; no validated aggregate is written. Individual
completed worker reports remain available for diagnosis, not an incomplete
selective comparison. Existing directories/results are never overwritten.

## Remaining full-runtime work

To call the comparison online query-to-answer, add and validate actual dense
query encoding/search, Exact MaxSim, SPLADE query encoding/search, and the legacy
approximate MaxSim branch required by CAPP. Their transitive checkpoint/index
availability and output equivalence are not established by graph replay alone.
Do not sum historical timers from different nodes or fold the 31-second replay
job duration into the online result. The current partial experiment is useful
evidence with explicit boundaries, not a replacement for those missing stages.

Local checks: nine new tests cover protected outputs, balanced order, condition
work, ranking failure gates, sanitized input bundles, paired aggregation,
hardware/cohort checks, GPU synchronization ordering (stubbed), and image-cache
policy. Seven graph-replay and eight cached-CAPP tests also pass. Bash syntax
passes. Live CUDA execution is necessarily pending HPC submission.
