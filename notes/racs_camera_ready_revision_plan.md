# RACS camera-ready revision worklist

Updated 2026-09-18. Working notes and proposed manual edits only; no manuscript
or Overleaf archive has been modified. The authoritative manuscript is the
September 15 ZIP, not the older `ACM_Paper` checkout.

Ready-to-paste edits for priorities 1 and 2 are in
[the manual revision guide](racs_manual_revisions_runtime_budget_control.md).
It gives exact manuscript anchors, three LaTeX tables, results text, and
protocol/limitations replacements. Apply manually; no manuscript was edited.

The expanded ablation table and BGE reader-job instructions are in
[the baseline/ablation guide](racs_manual_revisions_baselines_ablation.md).
The BGE QA numbers remain pending: HPC job 15911612 was last reported running
on gpu009. Its completion has not been checked locally.
Feature rationale and consolidated limitations are ready in
[the feature-rationale guide](racs_manual_revisions_feature_rationale.md), with
an [advisor/reviewer checklist](racs_reviewer_revision_checklist.md).
The [remaining execution plan](racs_remaining_execution_plan.md) records the
CPU-only prerequisite audit for a faithful pipeline benchmark and the
training-only 20% alpha-selection protocol located in the thesis.
Audit job 15911623 completed (0:0): all four graph artifacts match the gold qids
and their direct recorded paths exist. Its saved configurations distinguish
the Exact MaxSim/score-selection base from legacy dense/MMR auxiliary rankings;
the shared SPLADE summary names `naver/splade-cocondenser-ensembledistil`, not
v3. The legacy dense prediction exists but its same-stem summary is missing.
Full-report inspection and graph replay remain pending before pipeline timing.

## Status and priorities

| Priority | Review request | Status / next action |
| --- | --- | --- |
| 1 | R1: cost of CAPP and sensitivity to reader budget | Reader QA at k=1,2,8 complete; reuse original k=4. Cached-input CPU benchmark validated in job 15908770 (196.2 ms/query). End-to-end latency and incremental memory claims remain unverified. |
| 2 | R2: labeler/content-feature coupling and gold-page injection | Job 15910645 completed: 2188 matched questions, four pages each, EM 44.06 / F1 51.39. GPP/CAPP matched-subset F1: 42.34 / 44.54. Ready for scoped diagnostic write-up; not independent human-gold validation. |
| 3 | R1: stronger reranking baselines | Saved LambdaMART/BGE/monoT5 reports match the thesis. LambdaMART is LightGBM LambdaRank with 40 features. BGE/monoT5 rerank 1000 pages. BGE four-page reader job 15911612 is submitted; result pending. BGE blend-selection provenance remains unresolved. |
| 4 | R1/R2: systematic ablation and missing comparison row | Six ablation QA runs rescored on identical 2441 qids with four pages each; saved retrieval reports now confirm page@4/page@10 too. Manual expanded table is ready. Models use fixed alpha 0.40 and matching recorded main inputs/settings apart from feature subset. Literal auxiliary-source paths remain unaudited for source-bearing ablations. |
| 5 | R1/R2: feature rationale and lightweight-design trade-offs | Manual draft ready: corrected feature descriptions checked against code, conceptual LTR/rank-fusion references checked, explicit source-input and capacity/cost limitations, and content-aware naming clarification. Author review and Overleaf insertion remain. |

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

Subsequent user-pasted HPC summaries confirm that the auxiliary names are not
merely stale labels: the document-hyperlink run records mode
`hyperlink_citation`, weight 2.25, and mean 19.1864 directed document edges;
the page-hyperlink run records weight 0.25, `target_pages`, 2441 hyperlink qids,
and mean 724.0705 directed PDF-hyperlink edges. The legacy no-hyperlink source
records zero hyperlink weights/counts. These facts establish hyperlink-enabled
auxiliary inputs to the matching replay, not the size of their causal benefit.

The saved full model has `auto_tune_blend_alpha=false` and fixed alpha 0.40.
The author recalls comparing multiple alpha values before choosing 0.40.
This is consistent with a fixed final run, but does not establish which
selection split was used or recover an earlier training-only tuning run.

The written protocol has now been located explicitly in thesis `section3.tex`
(lines 314–318), `section4.tex` (230–234), and `section5.tex` (833–838): a
training-only 20% holdout selects alpha by page@4, followed by full-training
refit. The September 15 paper states the same protocol. This establishes what
the documents claim; the new prerequisite audit searches relevant saved
tuning records to check that history without treating current fixed-run
settings as evidence for or against a separate earlier tuning run.

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
and no runtime report was produced by that failed job. The corrected launcher
was committed as a4d125b; the retry succeeded as described below.

### Validated HPC result: job 15908770

Source: user-pasted report and Slurm accounting. Job completed with exit 0:0
in 00:33:12 on node031. The job duration includes preparation, validation,
one warmup and three measured passes; it is not a single-pass inference time.
The full report remains on HPC at
`output/racs_capp_runtime_15908770/runtime.json` (not copied locally).

Hardware: one CPU thread on Intel Xeon Gold 6342 @ 2.80 GHz. All 2,441 complete
rankings matched the saved artifact in each of the three measured passes.
Candidate pool: 1,000 pages/question; source labels: gpp_no_hyperlink,
gpp_doc_hyperlink, gpp_page_hyperlink. No model retraining was performed.

| Measurement | Observed value |
| --- | ---: |
| Mean cached-input CAPP latency | 196.225 ms/question |
| Median of per-question mean latency | 184.073 ms |
| p95 of per-question mean latency | 338.180 ms |
| Serial throughput | 5.096 questions/s |
| Mean computation time for a 2,441-question pass | 478.986 s |
| One observed file-loading/preparation pass | 50.395 s |
| Process peak RSS through benchmark | 5963.797 MiB (5.824 GiB) |

The median and p95 above describe each question's mean over three measured
passes, not pooled individual-request latency. Per-pass means were 195.888,
196.423 and 196.365 ms/question. Averaging the printed stage means:

- Candidate preparation: 1.178 ms/question.
- Feature extraction: 193.472 ms/question (approximately 98.6% of total).
- Standardization and logistic scoring: 0.368 ms/question.
- Blending and sorting: 1.207 ms/question.

Do not report 5.824 GiB as the model size or incremental CAPP-only memory:
it is the whole benchmark-process high-water mark, including inputs and
reference validation. Do not interpret 196.225 ms as total retrieval + reader
latency or compare it directly with historical GPU reader timing as a
controlled percentage overhead. Upstream source generation is excluded.

Proposed manual runtime paragraph:

> With cached upstream rankings and page-text features loaded, CAPP reranked
> 1,000 candidate pages per question in an average of 196.2 ms on one CPU thread
> of an Intel Xeon Gold 6342, corresponding to 5.10 questions/s. Measurements
> cover 2,441 questions over three passes following one warmup pass; every
> reranked page list matched the saved evaluation artifact. Query-dependent
> feature extraction accounted for approximately 98.6% of runtime, while
> standardization and logistic scoring required 0.37 ms per question. These
> measurements cover the CAPP stage with cached upstream inputs, not retrieval,
> auxiliary graph generation, or VLM answer generation.

If reporting memory/preparation in a table, use the explicit scope labels above.

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
imply half the image tokens, memory or latency. Use the validated CPU-stage
measurements with their stated boundaries; historical reader timings do not
establish a controlled GPU speedup.

## Pseudo-label control: a concrete gap in the submitted manuscript

The September 15 manuscript already reports pseudo-gold F1 55.82 on 2,188
labeled questions alongside CAPP F1 45.69 on all 2,441 questions. This is not a
matched-population causal comparison, even though the table states the counts.
Reusing that number alone would not add the control requested by Reviewer 2.

The user-provided HPC summary audit confirmed:

| Existing run | Label coverage | Evaluated qids | Context | EM / F1 |
| --- | ---: | ---: | --- | --- |
| adaptive_exactonly pseudo_gold_only | 2188 | 2188 | no base fill; 2146 contexts shorter than four pages | 47.989 / 55.817 |
| older evidence_countmatch pseudo_gold_only | 2285 | 1515 | no base fill; 1513 contexts shorter than four pages | 47.855 / 56.603 |
| older support_doc_non_gold_only | 2285 | 1514 | no base fill; 1512 contexts shorter than four pages | 26.486 / 33.120 |

The mean labeled-page count in the current oracle summary is 1.4785, but that
statistic is computed before truncation and must not be labeled the exact mean
number of reader images without checking the prediction file. The generic
`output/m3docvqa_pseudo_gold_reader_oracle` directory was absent. At the time
of that initial audit, no completed fixed-four-page injection was shown; the
new completed run is documented below. The older positive/negative
results should not be treated as a matched current-label control without
restricting to a verified common question set and confirming label provenance.

### Matched-question evaluation completed

The CPU-only audit verified that the existing oracle's 2,188 filtered question
IDs, pseudo-page label sets and reference answers match the current paper's
adaptive-exact labels. GPP and CAPP answers were reused from the original
four-page runs; all three methods cover every question in that cohort.

| Method | Questions | QA EM | QA F1 | Actual reader context |
| --- | ---: | ---: | ---: | --- |
| GPP | 2188 | 36.3346 | 42.3368 | four pages for every question |
| CAPP | 2188 | 38.1627 | 44.5416 | four pages for every question |
| Pseudo-gold only | 2188 | 47.9890 | 55.8172 | 1332 one-page, 775 two-page, 39 three-page, 42 four-page contexts |

CAPP improves over GPP by 2.20 F1 points on this matched subset. The oracle/CAPP
gap is 11.28 F1 points, but their page counts still differ; do not attribute
the whole gap solely to evidence correctness. Do not replace the paper's
full-2441 GPP/CAPP main results with these subset scores without relabeling.

### Fixed-four-page pseudo-gold injection: completed and validated

User-pasted accounting and final output confirm job 15910645 completed with
exit 0:0 in 02:15:54 on gpu008. The final `INJECTION_READER_RESULT` marker
confirms 2188 questions and four consumed reader pages each. EM is
44.05850091407678 and F1 is 51.38756855575869. No model retraining occurred.
The output directory is
`output/racs_adaptive_gold_injection_top4_15910645`; the evaluation file is
`mmqa_dev_pseudo_gold_plus_gpp_fill_qwen2vl_top4.eval.json`.

| Reader input | Matched questions | Pages/question | EM | F1 |
| --- | ---: | ---: | ---: | ---: |
| GPP | 2188 | 4 | 36.33 | 42.34 |
| CAPP | 2188 | 4 | 38.16 | 44.54 |
| Pseudo-gold injection + GPP fill | 2188 | 4 | 44.06 | 51.39 |

The injection gain is 6.85 F1 points over matched-subset CAPP and 9.05 over
matched-subset GPP. This controls question population and page count, but not
page order or candidate-pool membership, and does not eliminate labeler bias.
Do not compare the new 51.39 directly against all-2441 CAPP 45.69 as if they
used the same question population.

New launcher: `examples/sbatch_racs_gold_injection_top4.sh`.
It uses the existing input builder with the paper's adaptive-exact gold and
Exact MaxSim GPP no-hyperlink base. For each of the 2,188 labeled questions,
it takes up to four pseudo-gold pages in their existing label order, then fills
vacant slots with the highest-ranked distinct GPP pages. Cases with more than
four labeled pages are counted explicitly. This is a privileged-label diagnostic
(pseudo-gold pages can lie outside the original retrieved pool), not a deployable
retrieval method or a comparison that holds page order constant.

The input builder had an edge case: when its initial list already contained
four pages, `append_base_fill` could append a fifth page before stopping. The
reader would then truncate to four. The helper now caps the input and returns
immediately when the budget is full; tests cover full/overfull budgets, duplicate
pages, order preservation, and CLI cohorts with one through five pseudo pages.
Previously audited oracle runs used `fill_from_base=false`; they are unaffected.
No existing saved prediction or model has been modified.

The launcher requests one GPU, eight CPUs, 64 GB RAM and a six-hour limit. It
uses Qwen2-VL-7B-Instruct, 16-bit weights and four pages, with the existing reader
implementation and no new training. It validates the cohort, reference labels,
pseudo-gold prefix and four-page uniqueness before loading the reader, and
validates complete output coverage and the exact consumed pages after QA.
It saves a run manifest with the input and code SHA-256s, explicit reader
settings, and data/model-directory configuration for later provenance checks.
It prints GPU information when available; no Git executable is required on the
compute node. Only new job-specific outputs are written, under
`output/racs_adaptive_gold_injection_top4_JOBID/`. A pre-existing run directory
causes an error instead of overwriting or silently resuming it.

The completed experiment used the following launcher. This is a historical
command, not a request to resubmit the already successful control:

```bash
sbatch examples/sbatch_racs_gold_injection_top4.sh
```

The observed final log marker was `INJECTION_READER_RESULT`. Compare its
EM/F1 with the matched rows above, not the all-2441 headline numbers.
The control addresses population and
page-count confounds, but it still uses heuristic labels and does not by itself
establish independently annotated page truth or eliminate labeler coupling.

The existing oracle builder supports `--fill-from-base`, which places labeled
pages first and fills remaining slots from the base ranking. Its launcher
supports `RUN_GOLD_PLUS_BASE_FILL`, as well as support-document non-gold controls.
Inspect saved summaries and outputs first; do not rerun the launcher with its
defaults: its default gold is the older 2,285-question label set, not the paper's
2,188-question adaptive-exact label set.

Audit checklist for any future control variant (not outstanding work required
to repeat the completed injection):

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
User-provided saved reports now confirm these values at the printed precision,
the expected adaptive-exact evaluation gold and Exact MaxSim GPP no-hyperlink
base paths, and 2441 prediction qids. Source:
[HPC report](</Users/hoseinerfan/.codex/attachments/a9f9bb1c-dd8d-47ba-9b5c-ee2281b8c2da/pasted-text.txt>).
This is verification of saved reports, not new model inference.

The retained BGE result uses `bge-reranker-base`, candidate/rerank depth 1000,
alpha 0.20, max length 512 and max page characters 6000. The retained monoT5
result uses `monot5-base-msmarco-10k`, depth 1000, alpha 1.0, max length 512
and max page characters 4000. Both report 744 empty-text encounters, not 744
missing questions. Separate top-100 trials exist and are not the thesis rows.
LambdaMART reports selected alpha 0.45 and a tuning record with 4233 evaluated
labeled tuning questions. The subsequent model audit confirms candidate depth
1000, LightGBM backend, LambdaRank objective, automatic alpha tuning, and 40
features. This is not an identical-feature comparison with 30-feature CAPP.
BGE's alpha-selection split is not established by these reports.

The latest filename search found zero name-matched baseline QA reports; this
does not rule out differently named artifacts. The new launcher
`examples/sbatch_racs_bge_reader_top4.sh` reuses the retained BGE top-1000,
alpha-0.20 rankings for one four-page reader run. It checks the exact cohort,
per-question candidate-set agreement with GPP, metadata, and selected reader
pages, and writes only an isolated job directory. No retraining is required.
Job 15911612 has been submitted; its result is pending. LambdaMART/monoT5 matched
reader QA and full source-input comparability remain separate outstanding items.
See `notes/racs_manual_revisions_baselines_ablation.md` for the run instructions
and the ready-to-insert ablation revision.

## Feature ablations: proposed manual expansion of Table 8

The QA columns below have now been independently recomputed from saved answers
in CPU job 15911329: every variant covers exactly the same 2441 gold qids,
has four selected reader pages per question, and matches its saved evaluation.
The subsequent user-provided retrieval reports confirm page@4 and page@10 for
the six ablations on 2188 labeled questions, with 2441 prediction qids. This is
saved-report verification, not new retrieval inference. Full CAPP was verified
separately. The newly included no-rank variant is highest on page@10; the old
boldface on structure + content's page@10 must therefore be removed.

| Variant | page@4 | page@10 | QA EM | QA F1 |
| --- | ---: | ---: | ---: | ---: |
| Structure only | 0.7573 | 0.8844 | 38.71 | 44.75 |
| Structure + content | 0.7779 | 0.8935 | 39.04 | 45.14 |
| No rank | 0.7692 | 0.8958 | 38.92 | 45.04 |
| No source | 0.7719 | 0.8729 | 39.12 | 45.22 |
| No structure | 0.6508 | 0.8140 | 37.65 | 43.46 |
| No content | 0.7573 | 0.8702 | 38.59 | 44.69 |
| All features | 0.7715 | 0.8825 | 39.41 | 45.69 |

The six ablation model files were found in
`output/m3docvqa_content_aware_exact_maxsim_direct_exactonly_adaptive_norm05_feature_matrix`.
Their prefix is `mmqa_train_to_dev_content_aware_feature_` and their suffix is
`_base_exact_maxsim_gpp.model.json`. A subsequent saved-configuration audit
confirmed that all six named feature sets match their actual feature lists,
all use alpha 0.40, and all match the full model's recorded training/evaluation
gold, base-ranking and page-text paths. Other saved arguments differ only in
`feature_set`. No separate tuning summary is recorded in these final artifacts.

Feature counts are 6 (structure only), 21 (structure + content), 25 (no rank,
named `source_structure_content`), 26 (no source), 24 (no structure), and 15
(no content, named `rank_source_structure`). The structure-only,
structure+content and no-source models have no source features. Source-bearing
variants retain the four aggregate source features; their literal auxiliary-
source filenames are not established by this main-input-path comparison.
The no-source variant is an existing separately trained alternative, not an
interpretation of the original 30-feature full-CAPP results.

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
