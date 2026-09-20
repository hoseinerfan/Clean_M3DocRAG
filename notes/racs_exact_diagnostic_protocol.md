# RACS: four-question Exact MaxSim diagnostic

Prepared September 20, 2026. No Overleaf/manuscript or historical output changes.

## First diagnostic outcome and required correction

Job 15915129 completed. The current original runner and all four CPU-query
runtime-helper conditions produced identical page lists and scores on all four
questions. Two questions exactly matched the saved historical scores/rankings;
the other two first differed at ranks 11 and 13. All four retained the same top-4
page order, but the two failing questions had score differences across all
1,000 pages, with maximum absolute differences 0.1829566956 and 0.0720348358.
These are not grounds for relaxing the checks. The GPU-query control matched
zero complete historical rankings. No online runtime was validated.

**Diagnostic flaw:** both the nominal default-thread and one-thread conditions
actually used one thread. The imported `benchmark_capp_runtime.py` sets OMP,
MKL and other thread environment variables to one at import time. Consequently,
15915129 did not test different CPU thread counts. Its original/helper agreement
and inference-context comparisons remain useful under the observed one-thread
setting; it cannot rule out thread-count effects or recover historical settings.

The corrected diagnostic below explicitly sets eight versus one PyTorch CPU
threads, restores OMP/MKL/OpenBLAS variables to the declared condition before
loading the original runner, records PyTorch parallel-backend information, and
refuses to summarize results unless the effective counts are verified as 8/1.
Eight fits the existing eight-CPU allocation; it is a predeclared diagnostic
condition, not a claim that the historical run used eight threads. Old outputs
remain untouched. Submit one new diagnostic job with the same launcher after
pulling this correction, not the full online runtime job.

## Reason and scope

Online-runtime job 15915128 passed the first question's FAISS candidate-pool
check, then failed Exact MaxSim replay. Its 1,000-page candidate set matched, but
the first order difference was rank 11. No completed online timing exists.
The stored full-query, Exact MaxSim, page-batch-64 settings agree with the new
runtime reconstruction. Both production embedding loaders convert the stored
embeddings to bfloat16; the inspected document file is itself BF16. No cause is
assumed from these checks.

Before another full timing submission, run
`examples/sbatch_racs_exact_diagnostic.sh`. It tests the same four preselected
warm-up questions, not a quality-selected subset. It reads:

- `output/racs_online_runtime_15915128/capp.online.bundle.json`
- The original `m3docvqa_exact_maxsim_mmqa_dev` summary and its gold file
- GPU query embeddings saved by diagnostic 15915127

The preparation checks question text byte-for-byte (including whitespace),
the exact-scoring options, embedding name, complete 1,000-page reference lists,
and the previous diagnostic's four-question cohort. Selected original gold rows
and candidate pages are copied into a new diagnostic directory. No synthetic
gold labels are substituted. A read-only check through the mounted cluster
confirmed these real input texts/options/embedding names/cohorts agree.

## Predeclared comparisons

Two fresh subprocesses invoke the actual current
`run_visual_rerank_batch.main()` entry point with the original Exact MaxSim
arguments, pinned model names, all query tokens, 1,000 candidates, and batch 64:

1. Explicitly use eight CPU threads, within the requested eight-CPU allocation.
2. Explicitly use one CPU thread, as the runtime benchmark does.

The CPU encoder's actual query outputs are captured without replacing its
calculation. After the original runner has finished and saved its rankings,
each process separately encodes the four questions under outer inference mode.
This isolates the benchmark's outer context from the original encoder's inner
no-grad context. Extra encodings cannot affect the earlier original run.

A third, single-threaded subprocess calls the production runtime
`OnlineRetriever.dense_scores(..., approximate=False)` directly, without
initializing the full online retriever. It tests five fixed query sources:

- Original eight-thread CPU queries, ordinary no-grad path.
- Eight-thread CPU queries, outer inference mode.
- Original one-thread CPU queries, ordinary no-grad path.
- One-thread CPU queries, outer inference mode.
- The prior diagnostic's saved direct-GPU queries.

Query NPZ files are verified against their file hash, float32 embedding hash,
shape, token IDs, and token count before scoring. Float32 serialization preserves
the values used by the production scorer, which already converts queries to
float32. The GPU-query condition is explicitly a cached-input diagnostic, not
fresh online retrieval. All five conditions use the same saved candidate pool
and original scoring configuration. There is no automatic setting selection.

The complete diagnostic contains seven conditions × four questions = 28
ranking comparisons. The strict original order/score checks are unchanged.
Eight versus one thread are diagnostic hypotheses, not hyperparameter tuning
or evidence of the historical launch environment. Current original-runner code
is not claimed to be an archived historical checkout.

## Resource and output contract

One GPU, eight CPUs, 192 GB memory, one-hour allocation limit. These are requested
resources, not a promised runtime. There is no FAISS index load/search, SPLADE,
graph/CAPP, Qwen reader, training, or publishable timing measurement.

Only a new `output/racs_exact_diagnostic_JOBID` directory and Slurm logs are
written. Existing directories are refused. Worker stdout/stderr is kept in
`original_eight_threads.log`, `original_one_thread.log`, and `harness.log`.
The top-level `diagnostic.json` reports each condition's complete ranking/score
match count. Completion means comparisons were collected, not that replay passed.
On failure, read `failure.json` and the corresponding worker log.

Next after this job: inspect every condition and captured query comparison;
identify the cause before changing the runtime implementation. Do not retrain,
loosen tolerances, silently substitute a better-performing method, or resubmit
the unchanged full benchmark. Even a four-question match still requires full
online validation before latency can be reported.
