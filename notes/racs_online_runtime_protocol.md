# RACS: full online runtime completion protocol

Prepared September 18, updated September 19, 2026 after the first failed attempts.
**No validated online timing result exists yet.** The goal is to close R1.2,
not stop at a first draft.
Original paper/Overleaf files and old experiment outputs are not modified.

## Current action: use the author's existing local SPLADE files

Job 15911873 failed with exit 1:0 after 4:56 on gpu010. Its startup log now
confirms the saved IVF metric is L2, the quantizer metric is inner product, and
nprobe was set to 4. All 3,366 document embeddings and both ColPali replicas
loaded. It then failed constructing the SPLADE tokenizer, with a missing
vocabulary-file path (`None`) in offline Hub-name resolution. This does not
establish that the author lacks the files, and no retrieval replay/timing passed.

The author confirmed local files exist and requested using them. The benchmark
now requires an explicit local SPLADE checkpoint directory. A separate local
tokenizer directory is supported if needed. Both loaders receive
`local_files_only=True`; there is no Hub-name/cache fallback and no download.
The original recorded model ID remains `naver/splade-cocondenser-ensembledistil`
for index/provenance checks; a path override is not permission to change models.
Configuration/tokenizer files are hashed, weights are inventoried, and every
upstream ranking/order/score check remains mandatory.

File-layout checks run before reading large prediction artifacts. In each
worker, the local SPLADE tokenizer/model load before the FAISS index and corpus
embeddings, so a bad local path fails early. Matching vocabulary sizes is checked
but is not proof of token-ID/weight equivalence; complete retrieval replay is
still required. The Mac workspace has no model directory; the HPC folder path
must be supplied by the author, not guessed. Do not submit a new GPU job until
that path is known and the read-only local-file check succeeds.

## September 19 correction: preserve the saved FAISS search metric

The author supplied accounting for two attempts: 15911849 failed with exit 1:0
after 2:11 on gpu004; 15911871 failed with exit 1:0 after 2:05 on gpu010.
The first job's traceback/preflight report identifies the benchmark's incorrect
`Expected an inner-product FAISS index` guard. The second job's detailed log was
not supplied, so its precise failure cause is not independently confirmed.
Neither produced evidence of successful online timing.

The repository's `examples/run_indexing_m3docvqa.py` creates an IP quantizer but
calls `IndexIVFFlat(quantizer, d, ncentroids)` without the metric argument. The
[official FAISS constructor documentation](https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexIVFFlat.html)
specifies L2 as that argument's default. Separately, `RAGModelBase` uses the
loaded index for candidate-token search and recomputes candidate scores from
embedding dot products when the token table is supplied. An embedding-dot page
score therefore does not prove an inner-product IVF search metric. The supplied
exception establishes a non-IP index, not its numeric metric; the next run logs
that metric explicitly before loading the corpus.

The benchmark now accepts L2 or inner-product search on the existing index,
records the search and quantizer metrics separately, and preserves both. It
does not rebuild the index, overwrite its metric, reinterpret L2 distances as
page scores, or change candidate-order/score validation. Unsupported metrics
still fail. New regressions cover an L2 IVF index with an IP quantizer, an IP
index, and rejection of unsupported metrics/invalid search settings.

That correction was exercised by 15911873, as recorded above. Further upstream
replay checks still have to pass; it does not establish output equivalence.

## What this job measures

`examples/sbatch_racs_online_runtime.sh` runs
`scripts/benchmark_racs_online.py`, reusing the tested graph/reader worker.
It requests one GPU, eight CPUs, 192 GB RAM and a 12-hour limit. This is an
allocation limit, not a predicted completion time; queue time is unknown.
The implementation requires at least 40 GiB GPU memory and never substitutes
a quantized reader to fit a smaller device. Existing models are loaded offline.

The complete **warm-service, serial, batch-one** query timer includes:

1. ColPali query encoding, FAISS IVFFlat nprobe=4 token search, page aggregation.
2. Query encoding for page-local scoring and Exact MaxSim on 1,000 candidates.
3. For CAPP, the additional legacy 224-token approximate MaxSim route.
4. SPLADE query encoding/pruning and corpus postings search.
5. Main GPP graph; for CAPP, three auxiliary graphs, source maps and CAPP scoring.
6. PDF rendering/page selection, prompt construction and Qwen generation.

Explicit GPU synchronization surrounds the query and GPU retrieval stages.
Offline corpus embedding/index construction and page-text extraction are excluded.
Resident model/index loading and input preparation are separately reported.
No retrieved rankings or query embeddings are reused across questions. Static
corpus embeddings and indices are resident. There is no cross-question PDF
image cache. Shared work within one CAPP query is performed once where it can
serve both exact and approximate paths without changing their outputs.

This is not a cold-start, concurrent-load or network-service benchmark. Reciprocal
mean time is serial throughput, not measured throughput under concurrency.

## Reconstruction choices, not recovered historical commands

The audit identifies the exact dense output, shared sparse output and separate
legacy approximate dense output. The existing entry points do not encode every
historical launch choice in those outputs. The new manifest explicitly records:

- Local ColPali backbone `colpaligemma-3b-pt-448-base` plus adapter `colpali-v1.2`;
  full query tokens, no PAD-score removal; preserve the loaded FAISS search
  metric and separately recompute embedding-dot page scores.
- GPU baseline query encoder and CPU page-scoring query encoder, following the
  current baseline and visual-reranking code paths. Two encoder replicas remain
  resident. Both query encodings are charged to both methods; exact/legacy reuse
  the CPU encoding within a CAPP query. The memory report includes both replicas.
- Exact page batch size 64, as recorded in its summary. Approximate query-mean
  global-top-224 fp32 scoring is unbatched, consistent with the recovered legacy
  diagnostic. Its incomplete historical summary remains a provenance limitation.
  Diagnostic-only pruning reports are not generated inside serving-time scoring.
- `naver/splade-cocondenser-ensembledistil`, transformers backend, 32 query terms,
  max length 64, batch one. The old summary identifies the model/term budget but
  does not establish its historical batch size/max length.

These choices cannot be accepted just because they look plausible. They must
reproduce the saved outputs. If they do not, the job stops with a diagnostic;
we inspect the mismatch before choosing any change. No silent cached-input
fallback, relaxed validation, new model selection or retraining is performed.

## Validation and experimental design

- Same outcome-independent 128-question hash sample and four disjoint warm-up
  questions as the completed graph-to-answer benchmark.
- Initial CAPP upstream preflight covers FAISS, exact, approximate and SPLADE
  outputs on the four warm-up questions before the expensive reader passes.
- Four repeats/method, eight fresh sequential workers, balanced AB/BA order in
  one allocation. Every worker runs its own four-question warm-up.
- For **every** warm-up/measured query, compare all 1,000 ordered page IDs and
  scores for FAISS, exact, sparse and (for CAPP) approximate retrieval, then each
  graph. Scores use rtol=1e-6/atol=1e-8, with exact ordered IDs required. CAPP's
  complete final order must also match. Validation is outside the query timer.
- Same Qwen2-VL-7B-Instruct, 16-bit weights, prompt, generation settings and four
  unique pages. Timing-subset answers never replace full-development QA scores.
- Aggregate only if cohorts, hardware/software, reader, upstream input identity,
  and all checks agree. Timed partitions must reconcile to total latency.
- Report mean/median/p95 of per-question means over four repeats and per-pass
  summaries. The p95 is not the tail of all individual raw requests.
- CPU whole-process high-water RSS and GPU allocated/reserved peaks are reported.
  CPU peaks include initialization/transient loading allocations and references;
  GPU reserved peaks include the warm-up allocator cache. Neither is isolated
  incremental scorer memory. Full corpus retrieval resources are now resident.
- Input summaries/predictions and source code are hashed. Encoder/embedding file
  identity also records paths, sizes and modification times; JSON configs are
  hashed. Full checkpoint/embedding byte hashes are not claimed.
- OS filesystem cache and external node load remain uncontrolled. No statistical
  significance or universal speedup claim follows from this one-node experiment.

## Submit after pushing and pulling the new commit

First set `RACS_SPLADE_MODEL_DIR` to the actual existing HPC checkpoint path.
If the tokenizer is in a different directory, also set
`RACS_SPLADE_TOKENIZER_DIR`; otherwise it defaults to the model directory.
The paths have not yet been supplied. Do not paste an invented example path.

Check the files on the login node without GPU/model loading or report writes:

```bash
env/bin/python -B scripts/benchmark_racs_online.py \
  --check-local-splade \
  --splade-model-dir "${RACS_SPLADE_MODEL_DIR:?Set the existing local checkpoint path}" \
  --splade-tokenizer-dir "${RACS_SPLADE_TOKENIZER_DIR:-$RACS_SPLADE_MODEL_DIR}"
```

This checks file layout/fingerprints, not tokenizer execution or retrieval
equivalence. After the paths are confirmed and changes pulled, submit from the
HPC login terminal with those variables exported:

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
if [ "$(git branch --show-current)" = "codex/mmdocir-hpc-workflow" ]; then
  git pull --ff-only origin codex/mmdocir-hpc-workflow &&
  mkdir -p output &&
  sbatch examples/sbatch_racs_online_runtime.sh
else
  echo "Stop: unexpected branch."
  git branch --show-current
fi
```

Only `output/racs_online_runtime_JOBID/` and that job's stdout/stderr are written.
Existing output directories cannot be overwritten or resumed automatically.

After completion, replace `JOBID` with the actual numeric ID:

```bash
sacct -X -j JOBID --format=JobID,State,ExitCode,Elapsed,NodeList -P
tail -n 20 output/racs_online_runtime_JOBID.out
tail -n 30 output/racs_online_runtime_JOBID.err
```

If successful, `runtime.json` must say `validated_online_query_to_answer`.
If unsuccessful, share `failure.json` and any `*.failure.json` in that unique
directory. A preflight failure is **not** a runtime result and must not be used
to populate the paper's cost table. Further numerical/kernel-path diagnosis may
be necessary; passing local mocks does not establish HPC output equivalence.

## Prior result independently checked

The author ran the separate raw-record arithmetic audit for job 15911732 and
reported `INDEPENDENT_RUNTIME_CHECK_PASSED`: GPP 3.9422036204741744 s, CAPP
4.702915329242387 s, delta 0.7607117087682127 s / 19.296611286575633%.
This confirms the cached-retrieval report's aggregation, not another timing run
and not online end-to-end latency. Preserve that narrower result as supporting
evidence; do not relabel it while waiting for this job.
