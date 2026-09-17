# RACS camera-ready revision worklist

Updated 2026-09-16. Working notes and proposed manual edits only; no manuscript
or Overleaf archive has been modified. The authoritative manuscript is the
September 15 ZIP, not the older `ACM_Paper` checkout.

## Status and priorities

| Priority | Review request | Status / next action |
| --- | --- | --- |
| 1 | R1: cost of CAPP and sensitivity to reader budget | Reader QA at k=1,2,8 complete; reuse original k=4. CPU benchmark implemented and locally tested, not yet run on HPC. End-to-end latency and incremental memory claims remain unverified. |
| 2 | R2: labeler/content-feature coupling and gold-page injection | Existing pseudo-gold-only result is already in the submitted paper. Inspect saved input summaries, matched qids and fixed-budget injection/negative controls before claiming this request is addressed. |
| 3 | R1: stronger reranking baselines | Thesis has LambdaMART/BGE/monoT5 retrieval results. Verify saved configurations, reranking depth, tuning split, evaluation gold and QA availability before importing the table. |
| 4 | R1/R2: systematic ablation and missing comparison row | Thesis contains all four leave-one-family-out variants and QA for structure + content. Verify saved evaluation artifacts, then add QA columns and these rows manually. |
| 5 | R1/R2: feature rationale and lightweight-design trade-offs | Draft factual interpretation below; verify cited prior work before manuscript insertion. Do not imply content-only drives the improvement. |

## Verified evaluation configuration

The full replay documented in `notes/racs_capp_artifact_provenance.md` reproduces
all 2,441 complete rankings and all 2,441,000 probabilities within 1e-6.
This establishes a working saved-model evaluation configuration, not recovery
of the original training command or proof of retraining equivalence.

- Base: Exact MaxSim GPP no-hyperlink, top 1,000 pages.
- CAPP: saved 30-feature logistic model, fixed blend alpha 0.40.
- Auxiliary features: three legacy GPP outputs in
  `output/m3docvqa_gpp_hyperlink_node_ablation_mmr_target1`:
  no-hyperlink, document-hyperlink and page-hyperlink variants.
- No fourth standalone SPLADE source in the reproducing CAPP configuration.
  This does not remove sparse retrieval from upstream GPP.
- A blanket statement that the full method uses no hyperlink-derived signals
  is not supported by this recovered configuration.

## CPU benchmark

New files:

- `scripts/benchmark_capp_runtime.py`
- `examples/sbatch_racs_capp_runtime.sh`
- `tests/test_benchmark_capp_runtime.py`

The launcher pins the verified artifact paths explicitly; it does not source
the older wrapper's fallback configuration. It requests one CPU, 64 GB RAM,
two hours on `compute`, and no GPU. NumPy thread environment variables are set
to one before importing NumPy. It runs one full warmup and three measured
passes. Every complete page order must match the saved prediction on every
pass; otherwise the job fails without writing a successful report.

Reported measurements:

1. One observed preparation pass: model and question loading, base/source JSON
   loading, page-text tokenization, and source-map construction.
2. Per-question candidate preparation; query/page feature extraction;
   standardization and logistic scoring; blending and sorting.
3. Mean, median, nearest-rank p95 and throughput for each measured pass; a
   separate aggregate summarizes each question's mean across repeats.
4. Process peak RSS, CPU/node/software information and input/code SHA-256s.

Limits: the report is **cached-input CAPP CPU inference**, not full RAG latency.
It excludes original page-text extraction, dense/sparse retrieval, auxiliary
graph generation, training, reader inference, and prediction serialization.
Input-file cache state is uncontrolled; preparation must not be called a
cold-start result. Peak RSS includes cached inputs, parsing temporaries and
reference validation; it is not incremental CAPP-only memory. Do not add the
three measured-pass durations to represent one inference run.

After pulling this commit on HPC, from the repository root:

```bash
mkdir -p output
sbatch examples/sbatch_racs_capp_runtime.sh
```

The job prints `saved_report=output/racs_capp_runtime_JOBID/runtime.json`.
The output location is job-specific and existing reports cannot be overwritten.
Original models, predictions and source files are read-only. GPU results and
upstream GPP timings are not synthesized from this CPU benchmark.

Job 15908742 failed before Python started because Git was absent from the
compute-node PATH. The launcher now treats Git logging as optional (as the
Python report already did); code SHA-256s are still recorded. No inference ran
and no runtime report was produced by that failed job. Resubmit after pulling
the launcher fix; a fresh job ID gives a separate output location.

## Reader-budget evidence

All eight prediction files passed the 2,441-question count and identical-qid-set
checks. New merged evaluations are from job 15906481. The k=4 rows are original
results, not fresh reader reruns. EM/F1 are percentages; times are seconds.

| k | GPP EM | GPP F1 | CAPP EM | CAPP F1 | GPP mean reader time | CAPP mean reader time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32.65 | 37.91 | 32.61 | 37.89 | 1.332 | 1.323 |
| 2 | 35.60 | 41.00 | 35.80 | 41.56 | 2.132 | 2.081 |
| 4 | 37.69 | 43.47 | 39.41 | 45.69 | 3.398 | 3.355 |
| 8 | 37.81 | 44.04 | 39.16 | 45.73 | 7.621 | 10.102 |

`time_qa` includes document-image loading through the cache and reader answer
generation, but excludes retrieval, CAPP reranking and initial reader loading.
Job allocations confirm one GPU, eight CPUs and 64 GB per shard, but do not
identify GPU models. Do not claim a controlled speedup across these runs. The
k=8 CAPP reader time is higher than GPP's; its cause is not established.

### Proposed manual addition: Results, reader-budget sensitivity

> Keeping the saved retrieval rankings fixed, we varied the maximum reader
> budget over 1, 2, 4, and 8 pages. CAPP achieved QA F1 scores of 37.89, 41.56,
> 45.69, and 45.73, compared with 37.91, 41.00, 43.47, and 44.04 for GPP.
> The largest observed advantage was at four pages (+2.22 F1 points); at one
> page CAPP did not improve QA. CAPP with four pages exceeded GPP with eight
> pages by 1.65 F1 points. Increasing CAPP's budget from four to eight pages
> changed F1 by only +0.04 points and reduced EM from 39.41 to 39.16.

No significance claim is supported yet. Half the maximum page budget does not
imply half the image tokens, memory or latency. Add runtime prose only after
the controlled CPU benchmark and remaining comparability checks.

## Pseudo-label control: a concrete gap in the submitted manuscript

The September 15 manuscript already reports pseudo-gold F1 55.82 on 2,188
labeled questions alongside CAPP F1 45.69 on all 2,441 questions. This is not a
matched-population causal comparison, even though the table states the counts.
Reusing that number alone would not add the control requested by Reviewer 2.

The existing oracle builder supports `--fill-from-base`, which places labeled
pages first and fills remaining slots from the base ranking. Its launcher
supports `RUN_GOLD_PLUS_BASE_FILL`, as well as support-document non-gold controls.
Inspect saved summaries and outputs first; do not rerun the launcher with its
defaults: its default gold is the older 2,285-question label set, not the paper's
2,188-question adaptive-exact label set.

Required audit before a new run:

- Confirm label version, exact qid set, actual page counts and reader settings.
- Re-evaluate GPP and CAPP answers on the identical control qids; the existing
  QA evaluation script accepts a filtered gold file, so generation need not be
  repeated solely to obtain subset scores.
- Look for an already completed fixed-budget pseudo-gold-plus-base-fill run.
- If absent, prepare one targeted k=4 injection run with the paper's adaptive
  labels, explicitly reporting it as **pseudo-gold**, not human-verified gold.
- For a negative control, match question population and page count; pages not
  labeled by the heuristic may still contain valid evidence, so call them
  unlabeled/non-pseudo-gold pages rather than proven irrelevant pages.
- An oracle or injection test measures answer usefulness but does not eliminate
  heuristic-label bias. Retain that limitation; independent manual evidence
  validation would be a stronger additional check.

## Stronger baselines: verify before reuse

Thesis `section5.tex`, table `tab:standard-rerankers`, reports page@4:
GPP 0.6376; LambdaMART 0.6408; BGE 0.6705; monoT5 0.6609; CAPP 0.7715.
These values have not been newly reproduced in this revision task.

Inspect the saved baseline summaries before describing them as matched:
`gold`/`eval_gold`, `base_pred`/`eval_base_pred`, candidate count, rerank depth,
text truncation, model identity, blend weight and tuning split. The current BGE
and monoT5 launchers default to reranking 100 of 1,000 candidates; defaults do
not prove the historical run used that setting. Disclose actual scoring depth.
Check saved top-four QA results; a retrieval-only comparison is not a newly
verified end-to-end QA baseline. Do not copy baseline numbers without checking
the evaluation cohort and metric definition.

## Feature ablations: proposed manual expansion of Table 8

The thesis already records the following additional evidence. Verify the saved
evaluation files on HPC before treating these as revision-validated results.

| Variant | page@4 | QA EM | QA F1 |
| --- | ---: | ---: | ---: |
| Structure only | 0.7573 | 38.71 | 44.75 |
| Structure + content | 0.7779 | 39.04 | 45.14 |
| No rank | 0.7692 | 38.92 | 45.04 |
| No source | 0.7719 | 39.12 | 45.22 |
| No structure | 0.6508 | 37.65 | 43.46 |
| No content | 0.7573 | 38.59 | 44.69 |
| All features | 0.7715 | 39.41 | 45.69 |

`No content` is the thesis's `Rank + source + struct.` row. Add the GPP reference
row and QA columns to the manuscript's ablation table. This supplies the missing
structure + content QA comparison supporting the full-model choice.

Suggested interpretation: structure is the strongest individual family in this
dataset, but the full model has the highest observed QA F1. Structure + content
has higher page@4 while the full model has higher QA F1; do not equate the
pseudo-page hit metric with answer utility. Avoid causal or statistically
significant superiority claims without further testing.

## Feature rationale and limitations: manual-edit guidance

Explain the intended role of each family next to `tab:capp-features`:
rank features preserve retrieval confidence/order; source features summarize
agreement across auxiliary rankings; structure features encode document/page
localization; content features measure question/page lexical compatibility.
These are design motivations, not evidence that the feature set is optimal or
an exhaustive feature search was conducted. Attribute inherited design ideas
to verified prior-work citations, and separate them from this study's empirical
choices. Discuss dataset-specific structural regularities, heuristic-label
coupling, dependence on extracted text and lack of direct visual reasoning in
the logistic scorer. Computational claims must include feature costs, not only
the 30 weights and one bias.

The acceptance email states October 7, 2026 for camera-ready materials and author
registration. Share a consolidated first revision in the advisor's agreed
review window; no automatic submission or email has been authorized.
