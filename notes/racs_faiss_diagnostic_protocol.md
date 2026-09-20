# RACS: four-question candidate-pool diagnostic

Prepared September 19, 2026. This is debugging, not a new runtime or QA result.
The benchmark's existing acceptance gates and defaults are unchanged.

## Why this is needed

Job 15911874 loaded the local SPLADE checkpoint and dense assets, but failed
the first FAISS candidate-pool comparison, on `dd8033ba02db5b76418eb0cd8ebcda71`.
The original June 7 raw baseline was inspected through a read-only Innovator
mount. Its expected second page and score match the failure report:
`28c05bc44976658366aab347b369c1fa`, page 14, score 25.38038432598114.
The regenerated second page was `fd3c433b2fbec26ce0fe08060709b49f`, page 15,
score 12.788486421108246. This is not merely a score-tolerance failure.

Local and HPC `rag/base.py` and `retrieval/colpali.py` file hashes match. The HPC
editable package points to this repository's `src`. The raw output stores page
rankings and timings but not invocation settings. A bounded search has not
recovered the original launch log; this is not a claim that it does not exist.

## Fixed diagnostic design

Reuse the failed job's `capp.online.bundle.json`, including its four preselected
warm-up questions, baseline references and pinned local SPLADE paths. Do not
select additional questions based on retrieval quality.

For every question compare three encoder paths:

1. Direct GPU model, as used by the new benchmark, under inference mode.
2. Its CPU scoring-model replica, also under inference mode.
3. The GPU model after `Accelerator().prepare(model)`, under no-grad, mirroring
   the current baseline entry point's model preparation. Record effective
   precision/device; do not claim to recover the historical launch environment.

Each encoding performs one fresh FAISS search at nprobe 4. A diagnostic adapter
then supplies the **same hits and distances** to the production aggregation
routine in its two supported branches: embedding dot products, and raw returned
FAISS scores. The index's serialized metric is never changed. On the observed L2
index, the latter branch really does use raw L2 distances in the existing
descending aggregation; reproducing it would not establish that it is a sound
similarity rule. No automatic switch is allowed.

Inspect at most 64 stored vectors, from 16 evenly spaced IVF lists and up to
four offsets in each, against the current token table and mapped page IDs.
Use raw IVFFlat vector bytes and release list buffers after access. Do not create
a direct map, rebuild or write the index. A passing sample is not an exhaustive
index-alignment certificate. API references:
[IVFFlat vector storage](https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexIVFFlat.html),
[inverted-list access and release](https://faiss.ai/cpp_api/struct/structfaiss_1_1InvertedLists.html).

Preserve the production order/score checks (1,000 pages, relative tolerance
1e-6, absolute tolerance 1e-8). Report candidate overlap, strict matches, first
order mismatch, and leading pages. Save query token IDs, summaries, float32
embedding NPZ files and all candidate rows in a fresh diagnostic directory.

## Scope and outputs

- No training, reader inference, graph/CAPP scoring, sparse retrieval or timing.
- The common loader still initializes local SPLADE and both ColPali replicas;
  this avoids changing the benchmark's asset loading while debugging.
- One GPU allocation, 8 CPUs, 192 GB RAM, one-hour allocation limit. Queue and
  actual execution time are not guaranteed. No extra dependency downloads.
- Reads only original assets; writes only a new `output/racs_faiss_diagnostic_JOBID`
  directory plus new Slurm stdout/stderr files.
- `diagnostic_complete_not_a_runtime_result` means all comparisons were collected,
  **not** that any ranking matched. Inspect all six totals and alignment results.
- Candidate replay is not proof of the historical launch command. Even if one
  branch matches all four, inspect the cause and then require the unchanged full
  online validation before publishing latency.

Submit from the HPC checkout after pulling the tested commit:

```bash
sbatch examples/sbatch_racs_faiss_diagnostic.sh
```

Default input is `output/racs_online_runtime_15911874/capp.online.bundle.json`.
No need to re-export the SPLADE path: this bundle contains the validated local
checkpoint/tokenizer location. Changing inputs requires an explicit
`RACS_DIAGNOSTIC_BUNDLE` override; never overwrite the previous bundle.
