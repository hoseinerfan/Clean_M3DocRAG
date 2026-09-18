# Controlled graph-to-answer timing: scope and execution

Updated 2026-09-18 after job 15911732 completed. Original paper/Overleaf and saved predictions are unchanged.
This experiment is **not full online query-to-answer runtime** and does not
by itself finish reviewer R1.2. It is the controlled downstream comparison
now completed using the replayed graph configurations and existing reader.

## Completed measurement: job 15911732

Accounting: `COMPLETED|0:0|01:23:12|gpu009`. The author supplied stdout, stderr
tail and extracted report fields in
`/Users/hoseinerfan/.codex/attachments/7b8bc269-b11b-4862-baac-93a20ac3524d/pasted-text.txt`.
The reported status is `validated_cached_retrieval_graph_to_answer`: 128 measured
questions, four repeats per method, NVIDIA A100 80GB PCIe. The program's success
gate requires matching graph orders/scores, matching complete CAPP orders, and
consistent cohorts/hardware/reader/software. These checks are reported by the
HPC program, not independently rerun locally against the raw per-query files.
The full report remains on HPC at
`output/racs_graph_reader_runtime_15911732/runtime.json`.

| Measurement | GPP | CAPP |
| --- | ---: | ---: |
| Mean graph-to-answer seconds | 3.9422036205 | 4.7029153292 |
| Median seconds | 3.5649435821 | 4.2865021792 |
| p95 seconds | 5.9405987030 | 7.9158358886 |
| Serial questions/second | 0.2536652330 | 0.2126340642 |
| Mean ranking seconds | 0.0980 | 0.6049 |
| Mean image-preparation seconds | 1.4509 | 1.7170 |
| Mean reader prompt/preprocess/generation seconds | 2.3933 | 2.3810 |
| CPU whole-worker peak RSS range, GiB | 4.9857–5.4267 | 6.4532–6.8022 |
| GPU peak allocated, GiB, every worker | 19.3656 | 19.3656 |
| GPU peak reserved, GiB, every worker | 24.7188 | 24.7188 |
| Qids with answer variation across repeats within method | 0 | 0 |

Latency quantiles summarize per-question means over repeats. CPU ranges are
the min/max of four fresh-worker high-water marks, not confidence intervals.
Every worker's CPU peak after initialization already equals its whole-worker
peak. These figures therefore cannot isolate active-query or scorer-only memory.
Equal GPU peaks in this workload do not prove equal memory for other budgets,
inputs or online deployments. CPU and GPU figures are separate, not additive.

Independently recomputed from the pasted high-precision means:
`4.702915329242387 - 3.9422036204741744 = 0.7607117087682127` seconds,
or `19.2966112866%` relative to GPP. The rounded stage differences reconcile:
`0.5069 + 0.2661 - 0.0123 = 0.7607` seconds. This is the extra time for the
measured graph-to-answer condition, **not** CAPP-only scorer cost or full online
overhead. Preparing different selected pages contributes to the measurement.
The three CAPP auxiliary graphs average approximately 0.3206 s together, source
maps 0.0051 s, and the CAPP stage 0.1825 s (feature extraction 0.1799 s).
Do not add nested stage totals again to the overall ranking total.

Pass means are GPP [3.9917, 3.8940, 3.9840, 3.8991] s and CAPP
[4.6847, 4.7794, 4.6766, 4.6710] s. No answer changed across repeats within
either method; this does not mean the methods produced identical answers or
establish their answer accuracy on this subset. Initialization/processor warnings
in stderr did not abort the job; this result does not establish that different
processor/attention versions would be equivalent.

Validation assessment: **share with explicit scope caveats** for the advisor
draft. The supplied summaries reconcile and the program reports all gates
passed. Preserve the raw JSON/worker reports for a future per-query audit;
there is no need to rerun this completed partial experiment solely for reporting.
No full online runtime, significance test, or new full-cohort QA result follows.

## Completed precheck

The author supplied successful accounting/stdout for graph replay job 15911730:
`COMPLETED|0:0|00:00:31|node011`. Main Exact-MaxSim/no-hyperlink GPP and each
of the three legacy auxiliary branches matched all 16 sampled questions:
candidate sets, complete 1000-page orders, and scores (rtol 1e-6, atol 1e-8).
These are 16 distinct questions checked in four branches, not 64 questions.
HPC report: `output/racs_graph_replay_15911730/replay.json`.
Its full JSON has not been transferred locally; benchmark job 15911732 read it
directly and passed the audit/input fingerprint gates before generating reports.

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

## Submission reference (already completed; do not submit a duplicate)

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
passes. Live CUDA execution completed in job 15911732; the reported results and
limits of local verification are recorded above.
