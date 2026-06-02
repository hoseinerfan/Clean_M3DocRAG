# MMQA Retrieval Scoreboard

Purpose: keep a short, updateable MMQA scoreboard with the exact tables, configs, artifact paths, and commands used to produce them. Extend this same structure in future chats and mirror it for other datasets.

## Metric Policy

- Use **average recall@k** as the primary retrieval metric everywhere.
- Keep **doc hit@k** as a secondary diagnostic only.
- Compare against the M3DocRAG baseline using the same recall-style metric as the baseline evaluator in `src/m3docrag/datasets/m3_docvqa/evaluate.py`.

## Method Labels Used Below

- `baseline`
  - current raw M3DocRAG retrieval baseline from the stronger saved `ret1000full_nprobe4` artifact
  - this is **not** the older historical April 10 `rag_dev_ret4` baseline
- `MaxSim+`
  - `plain_top224`
- `Graph Page Preserve`
  - page-preserving Graph-PPR using `plain_top224` dense + SPLADE sparse
  - the tables below use profile `denseheavy_lightboth`
- `Graph Page Preserve / page-labeled`
  - page-preserving Graph-PPR for datasets with exact gold page labels
  - current single general profile: `denseheavy125_medium_both`
  - use this same frozen config in the tables; if a dataset has no recorded value for this config yet, write `N/A`
- `Safe Heading/Bodyguard Gate`
  - precision-oriented rescue layer on top of heading-augmented graph views
  - accepts only narrow rank-window page promotions with multi-view heading support, displaced-boundary heading comparison, layout-query abstention, and body-evidence guard
  - report as a conservative post-processing gate, not as a global reranker
  - default runner profile is cutoff-relative `boundary`: for `HIT_K=k`, it may rescue only base rank `k+1` into top-`k`; all recorded results below use `HIT_K=4`
  - use `SAFE_GATE_PROFILE=window20` for the exploratory fixed rank `5-20` scan used with the recorded top-4 studies

## Table A: Doc Hit@k

These are document hit-rate numbers, not recall. They are still useful as a shortlist diagnostic.

| Method | Whole Dev doc@4 | Whole Dev doc@20 | ImageListQ doc@4 | ImageListQ doc@20 |
| --- | ---: | ---: | ---: | ---: |
| baseline | `2193 / 2441 = 89.84%` | `2340 / 2441 = 95.86%` | `62 / 141 = 43.97%` | `91 / 141 = 64.54%` |
| MaxSim+ | `2277 / 2441 = 93.28%` | `2368 / 2441 = 97.01%` | `81 / 141 = 57.45%` | `101 / 141 = 71.63%` |
| Graph Page Preserve | `2346 / 2441 = 96.11%` | `2401 / 2441 = 98.36%` | `86 / 141 = 60.99%` | `114 / 141 = 80.85%` |

## Table B: Average Recall@k

These are the main retrieval numbers to compare with the baseline evaluator.

| Method | Whole Dev recall@4 | Whole Dev recall@20 | ImageListQ recall@4 | ImageListQ recall@20 |
| --- | ---: | ---: | ---: | ---: |
| baseline | `71.56%` | `84.72%` | `32.27%` | `58.60%` |
| MaxSim+ | `74.89%` | `86.53%` | `43.91%` | `62.68%` |
| Graph Page Preserve | `76.18%` | `89.29%` | `42.02%` | `67.88%` |

Interpretation:

- Whole MMQA dev:
  - `Graph Page Preserve` is best at both `recall@4` and `recall@20`.
- `ImageListQ`:
  - `MaxSim+` is slightly better at `recall@4`.
  - `Graph Page Preserve` is clearly better at `recall@20`.

## M3DocVQA GPP Hyperlink Ablations 2026-05-29

Detailed interpretation note: [graph_ablation_findings_2026-05-28.md](/Users/hoseinerfan/Desktop/Clean_M3DocRAG/notes/graph_ablation_findings_2026-05-28.md:1)
Focused M3DocVQA note: [m3docvqa_gpp_ablation_findings_2026-05-29.md](/Users/hoseinerfan/Desktop/Clean_M3DocRAG/notes/m3docvqa_gpp_ablation_findings_2026-05-29.md:1)

M3DocVQA has document-only gold in this setup, so `doc@k` is primary and `row@k` is a returned-row diagnostic rather than exact page-gold recall.

Latest focused MMR hyperlink-node rows:

| Method | doc@1 | doc@4 | doc@10 | doc@20 | row@4 | row@10 | row@20 | Finding |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| no hyperlink MMR | 0.608 | 0.846 | 0.903 | 0.927 | 0.788 | 0.849 | 0.893 | Current MMR baseline. |
| `docnode_to_hyperlink_docs` + `log_count` | 0.608 | 0.851 | 0.907 | 0.930 | 0.798 | 0.855 | 0.898 | Conservative hyperlink gain, preserves top-1. |
| `pagenode_to_hyperlink_pages` + `log_count`, target pages 4 | 0.604 | 0.854 | 0.908 | 0.933 | 0.768 | 0.857 | 0.899 | Best doc@4, but row@4 drops. |
| `pagenode_to_hyperlink_pages` + `log_count`, target pages 1 | 0.600 | 0.853 | 0.911 | 0.936 | 0.804 | 0.864 | 0.905 | Best current context/deeper recall default. |

Dev-split doc-fusion probe:

| Method | eval doc@4 | Delta vs best source | Finding |
|---|---:|---:|---|
| best single eval source | 0.8518 | 0.0000 | Best individual source on eval split. |
| tuned doc fusion | 0.8613 | +0.0094 | Strong green-light result for doc-level tuning. |

Best tuned weights: dense `2.0`, SPLADE `4.0`, no-hyperlink GPP `0.0`, doc-hyperlink GPP `0.25`, page-hyperlink GPP `4.0`.

Older complete-wrapper and doc-doc rows remain useful for historical comparison:

| Method | doc@1 | doc@4 | doc@10 | row@4 | row@10 | Finding |
|---|---:|---:|---:|---:|---:|---|
| `docdoc_no_doc_doc` / score baseline | 0.608 | 0.846 | 0.903 | 0.758 | 0.848 | Score-only baseline for the older complete ablation. |
| `docdoc_hyperlink_citation`, tuned `w2.25` | 0.608 | 0.851 | 0.907 | 0.770 | 0.854 | Best older doc-doc hyperlink weight. |
| `select_mmr_docdiv_pool20_b0p10` | 0.608 | 0.846 | 0.903 | 0.799 | 0.850 | Older final-selection-only row@4 gain. |
| `docseed_page_sum_0p50` | 0.609 | 0.847 | 0.903 | 0.756 | 0.845 | Tiny doc@4 gain, worse row@4; not a main setting. |

Current M3DocVQA recommendations:

- Use `pagenode_to_hyperlink_pages + log_count + PDF_HYPERLINK_TARGET_PAGES_PER_DOC=1 + MMR doc-diverse` as the current best GPP hyperlink config.
- Use `docnode_to_hyperlink_docs + log_count + MMR doc-diverse` as the conservative alternative when preserving top-1 matters.
- Treat the dev-split doc-fusion result as the strongest signal that train-set document-level tuning is worth doing next.
- Do not use doc-seed page-score as a main M3DocVQA setting; it is weaker than hyperlink-node propagation.
- Keep external hyperlink claims scoped to M3DocVQA until OpenDocVQA/ViDoRe/DUDE/ViDoSeek/SciEGQA have dataset-internal URL-to-doc mappings.
- M3DocVQA train has gold docs but no gold pages, so train-set tuning should be document-level unless pseudo page labels are created.

## M3DocVQA Pseudo-Page Evidence Promotion 2026-06-02

Detailed interpretation note: [page_evidence_promotion_findings_2026-06-02.md](/Users/hoseinerfan/Desktop/Clean_M3DocRAG/notes/page_evidence_promotion_findings_2026-06-02.md:1)

New pseudo-page labels were created from MMQA evidence and exported page text. The strict setting matched `2,285 / 2,441` dev qids (`0.9361`) and `22,389 / 23,817` train qids (`0.9400`). These labels make page-level evidence retrieval measurable on M3DocVQA/MMQA.

Main strict pseudo-page dev result:

| Method | page@4 | page@5 | Gain@5 vs dense | page@10 | doc@4 | Finding |
|---|---:|---:|---:|---:|---:|---|
| Dense baseline | 0.6210 | 0.6556 | +0.0000 | 0.7383 | 0.9160 | Dense page pool baseline. |
| GPP no-hyperlink | 0.6740 | 0.7221 | +0.0665 | 0.8328 | 0.9488 | Strong graph baseline without hyperlink propagation. |
| GPP doc-hyperlink | 0.6687 | 0.7138 | +0.0582 | 0.8280 | 0.9514 | Hyperlink helps doc recall, not page evidence here. |
| GPP page-hyperlink | 0.6709 | 0.7177 | +0.0621 | 0.8298 | 0.9510 | Similar page-level behavior to doc-hyperlink. |
| Content-aware promotion, adaptive `page@5` | 0.7659 | 0.8031 | +0.1475 | 0.8687 | 0.9545 | Best current page-evidence method. |

The adaptive `page@5` method is a real held-out train tuning step. It chose `blend_alpha=0.40` by optimizing pseudo-page `page@5`, then retrained on all train qids before dev evaluation.

Counterfactual page promotion is a safety-oriented extension. It learns whether a candidate page should be inserted to repair the current top-k evidence set rather than reranking the whole list.

| Method | page@4 | page@5 | page@10 | doc@4 | doc@5 | Finding |
|---|---:|---:|---:|---:|---:|---|
| GPP no-hyperlink base | 0.6740 | 0.7221 | 0.8328 | 0.9488 | 0.9540 | Base for the counterfactual run. |
| Counterfactual page promotion, insert rank 5 | 0.6740 | 0.7707 | 0.8530 | 0.9497 | 0.9584 | Safe top-5 repair; preserves top-4. |
| Counterfactual page promotion, insert rank 4 | 0.7282 | 0.7694 | 0.8556 | 0.9510 | 0.9580 | Stronger top-4 repair; `196` recovered, `72` lost, net `+124`; threshold `0.80`. |
| Rank-4 gain | +0.0543 | +0.0473 | +0.0228 | +0.0022 | +0.0039 | Improves early page evidence without hurting document recall. |

Current interpretation:

- Main method: content-aware promotion with adaptive `page@5` tuning.
- Targeted repair extension: counterfactual page promotion. The insert-rank-4 variant now gives a real `page@4` improvement, while insert-rank-5 remains the conservative top-5 repair.
- Standard LightGBM LambdaMART was tested as a control. It improved `page@4` over GPP no-hyperlink (`0.6954` vs `0.6740`) but hurt `page@10` and doc recall, so it is not the main method.
- The strongest claim is pseudo-page-supervised, graph-aware content page promotion, not generic LTR.

## Table C: External Page-Labeled Method Tables

Use this section for datasets with exact page labels. These numbers are **average page/doc recall@k**, not answer EM/F1. Raw dense baseline rows are not recorded here unless explicitly listed; the complete rows we currently have are `MaxSim+` (`plain_top224`) and page-preserving Graph-PPR.

### MMDocIR

| Method | Page recall@4 | Page recall@20 | Doc recall@4 | Doc recall@20 |
| --- | ---: | ---: | ---: | ---: |
| MaxSim+ | `60.75%` | `74.80%` | `80.58%` | `88.90%` |
| Graph Page Preserve (`denseheavy125_medium_both`) | `67.19%` | `78.89%` | `81.60%` | `89.38%` |

### SciEGQA-Bench

| Method | Page recall@4 | Page recall@20 | Doc recall@4 | Doc recall@20 |
| --- | ---: | ---: | ---: | ---: |
| MaxSim+ | `73.94%` | `87.58%` | `90.70%` | `97.72%` |
| Graph Page Preserve (`denseheavy125_medium_both`) | `81.52%` | `92.48%` | `92.91%` | `98.71%` |

### ViDoRe V3

| Method | Page recall@4 | Page recall@20 | Doc recall@4 | Doc recall@20 |
| --- | ---: | ---: | ---: | ---: |
| MaxSim+ | `33.12%` | `54.31%` | `88.54%` | `98.09%` |
| Graph Page Preserve (`denseheavy125_medium_both`) | `64.65%` | `82.27%` | `90.99%` | `97.88%` |

Latest selected ablation finding: doc-doc edges did not improve ViDoRe average page recall. Doc-seed RRF gives only tiny average page recall gains and hurts doc recall at higher weights. Hard cross-doc selection (`max1doc`) is harmful.

### ViDoSeek

| Method | Page recall@4 | Page recall@20 | Doc recall@4 | Doc recall@20 |
| --- | ---: | ---: | ---: | ---: |
| MaxSim+ | `89.58%` | `98.42%` | `99.82%` | `100.00%` |
| Graph Page Preserve (`denseheavy125_medium_both`) | `N/A` | `N/A` | `N/A` | `N/A` |

### OpenDocVQA

| Method | Page recall@4 | Page recall@20 | Doc recall@4 | Doc recall@20 |
| --- | ---: | ---: | ---: | ---: |
| MaxSim+ | `51.22%` | `65.99%` | `53.07%` | `69.55%` |
| Graph Page Preserve (`denseheavy125_medium_both`) | `58.63%` | `76.62%` | `60.35%` | `79.22%` |

Latest selected ablation finding: `fully_connected_topdocs` is a useful non-hyperlink graph method for OpenDocVQA, improving average page recall@4 from `0.5863` to `0.6003` and average doc recall@4 from `0.6035` to `0.6189`. Doc-seed variants hurt OpenDocVQA and should not be used as the main setting.

### MMLongBench DocQA

| Method | Page recall@4 | Page recall@20 | Doc recall@4 | Doc recall@20 |
| --- | ---: | ---: | ---: | ---: |
| MaxSim+ | `N/A` | `N/A` | `N/A` | `N/A` |
| Graph Page Preserve (`denseheavy125_medium_both`) | `N/A` | `N/A` | `N/A` | `N/A` |

### DUDE

| Method | Page recall@4 | Page recall@20 | Doc recall@4 | Doc recall@20 |
| --- | ---: | ---: | ---: | ---: |
| Dense baseline (`baseline_ret1000`) | `53.54%` | `65.44%` | `61.14%` | `72.48%` |
| MaxSim+ (`plain_top224_ret1000`) | `57.20%` | `68.66%` | `65.07%` | `75.61%` |
| Dense+SPLADE doc-RRF | `53.12%` | `61.80%` | `66.31%` | `77.47%` |
| Graph Page Preserve (`denseheavy125_medium_both`) | `58.82%` | `71.72%` | `67.96%` | `78.33%` |

Interpretation:

- MMDocIR, SciEGQA-Bench, and ViDoRe V3 currently have recorded results for the single page-labeled default: `denseheavy125_medium_both`.
- ViDoSeek is almost saturated under `plain_top224`; the previously recorded dataset-specific best used heavier graph weights, but this table intentionally leaves the uniform `denseheavy125_medium_both` row as `N/A` until that exact config is recorded.
- OpenDocVQA now has a valid OCR-backed Graph-PPR row. The earlier all-empty SPLADE run should still be ignored.
- MMLongBench DocQA now has a converted page-labeled split, but retrieval numbers are not available until embeddings, dense retrieval, `plain_top224`, SPLADE, and Graph-PPR finish.
- DUDE is prepared and now has dense baseline, `plain_top224`, SPLADE/doc-RRF, and Graph-PPR results. Graph-PPR is best on DUDE at page@4/page@20 and doc@4/doc@20.

## Table D: Safe Heading/Bodyguard Gate

These numbers are page hit counts at `k=4`, not average recall. The gate is intentionally conservative; zero-loss behavior is more important than large acceptance. ViDoSeek and DUDE rows marked as audits were produced with dataset-specific restrictions that are no longer enabled by default in EvidenceGuard-PPR.

| Dataset | accepted | base page hit@4 | candidate page hit@4 | gated page hit@4 | recovered | lost | net | body rejects |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MMDocIR | 38 | 1114 | 1113 | 1117 | 3 | 0 | +3 | 25 |
| MMDocIR (`native codeguard` ablation; rejected) | 26 | 1114 | 1117 | 1114 | 1 | 1 | 0 | 25 |
| SciEGQA-Bench | 28 | 1323 | 1328 | 1323 | 0 | 0 | 0 | 23 |
| SciEGQA-Bench (`pymupdf4llm==0.3.4`) | 2 | 1323 | 1323 | 1324 | 1 | 0 | +1 | 5 |
| ViDoSeek (`page-0-block` audit) | 45 | 1023 | 1033 | 1029 | 6 | 0 | +6 | 30 |
| ViDoSeek (`pymupdf4llm==0.3.4`, `page-0-block` audit) | 51 | 1023 | 1019 | 1029 | 6 | 0 | +6 | 16 |
| DUDE (`doc-rank-1` audit) | 6 | 1733 | 1730 | 1733 | 0 | 0 | 0 | 1 |
| ViDoRe V3 (`text-heading` no-op) | 0 | 9383 | 9383 | 9383 | 0 | 0 | 0 | 0 |

Safe-gate artifacts:

- MMDocIR summary:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/MMDocIR_M3DocRAG/output/mmdocir/heading_breadcrumb_pdf_markdown_source_ablation/mmdocir_heuristic_strict_safe_gate_bodyguard.summary.json`
- MMDocIR native codeguard ablation summary:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/MMDocIR_M3DocRAG/output/mmdocir/heading_breadcrumb_pdf_markdown_source_ablation/mmdocir_safe_gate_bodyguard_codeguard.summary.json`
- SciEGQA-Bench summary:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/SciEGQA_M3DocRAG/output/sciegqa/heading_breadcrumb_pdf_markdown_source_ablation/sciegqa_safe_gate_bodyguard.summary.json`
- SciEGQA-Bench PyMuPDF4LLM summary:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/SciEGQA_M3DocRAG/output/sciegqa/heading_breadcrumb_pdf_markdown_pymupdf4llm_source_ablation/sciegqa_safe_gate_bodyguard.summary.json`
- ViDoSeek summary:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/ViDoSeek_M3DocRAG/output/vidoseek/heading_breadcrumb_pdf_markdown_source_ablation/vidoseek_strict_support_gate_layoutblock_no_page0_bodyguard.summary.json`
- ViDoSeek PyMuPDF4LLM summary:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/ViDoSeek_M3DocRAG/output/vidoseek/heading_breadcrumb_pdf_markdown_pymupdf4llm_source_ablation/vidoseek_safe_gate_bodyguard_no_page0.summary.json`
- DUDE summary:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/DUDE_M3DocRAG/output/dude/heading_breadcrumb_pdf_markdown_source_ablation/dude_safe_gate_bodyguard_docrank1.summary.json`
- ViDoRe V3 summary:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/ViDoRe_M3DocRAG/output/vidore-v3/heading_breadcrumb_text_source_ablation/vidore_safe_gate_bodyguard.summary.json`

ViDoRe note: the text-derived Markdown variants contain `0` raw outline headings and `0` raw
heuristic headings, so the heading-augmented graph views are identical to the no-heading control.
This is a valid no-op transfer result for the heading gate, not evidence that heading rescue failed
when headings are present.

MMDocIR native codeguard note: codeguard removed `69` heuristic heading lines on `101` code-dense
pages, but its gate output fell to `1114` page hits at `@4` with `1` recovered and `1` lost.
It is a negative ablation and is not part of EvidenceGuard-PPR.

Pending safe-gate evaluation targets:

```bash
DATASETS="m3docvqa" \
bash examples/run_safe_heading_gate_selected_datasets.sh
```

Rank-window variant:

```bash
SAFE_GATE_PROFILE=window20 \
RUN_GOLD_RANK_AUDIT=1 \
DATASETS="m3docvqa" \
bash examples/run_safe_heading_gate_selected_datasets.sh
```

The window profile writes `*_safe_window20_gate_bodyguard*.summary.json` plus optional
`*.gold_rank_positions.{json,md}` audits. The audit now reports first gold document ranks and
page/doc rank-band matrices, which is the main check for whether rank `6-20` misses are
same-document/page-local opportunities or document-retrieval failures.

Adaptive boundary example:

```bash
HIT_K=8 \
SAFE_GATE_PROFILE=boundary \
RUN_GOLD_RANK_AUDIT=1 \
DATASETS="mmdocir" \
bash examples/run_safe_heading_gate_selected_datasets.sh
```

This writes `*_safe_gate_bodyguard_top8.*` and can promote only a base rank-9 page into top 8.

Pending top-8 gate-policy ablation, after the native adaptive boundary run completes:

```bash
HIT_K=8 \
PDF_MARKDOWN_BACKEND=native \
DATASETS="mmdocir sciegqa vidoseek dude" \
bash examples/run_safe_gate_policy_ablation_selected_datasets.sh
```

This reads the existing graph predictions and produces
`output/safe_gate_policy_ablation/native_boundary_top8_policy_ablation.md`, comparing the control
document-rank cap against disabled and strict top-k-document variants plus relaxed agreement,
without overwriting the primary gate output.

After the boundary and policy runs, audit the case-level failures and recoveries:

```bash
HIT_K=8 \
DATASETS="mmdocir sciegqa vidoseek dude" \
bash examples/audit_safe_gate_topk_boundary_cases.sh
```

Paste `output/safe_gate_top8_case_audit/safe_gate_top8_case_diagnostics.md` rather than the full
logs.

Current M3DocVQA safe-gate input paths:

```text
M3DOCVQA_DENSE_PRED=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json
M3DOCVQA_SPARSE_PRED=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_splade_mmqa_dev/mmqa_dev_splade.prediction.json
M3DOCVQA_PAGE_TEXT_JSONL=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_page_text/m3docvqa_dev_page_text.jsonl
```

Use the raw `mmqa_dev_splade.prediction.json`; graph/source-ablation/no-SPLADE artifacts should not
be used as the sparse input for this run.

M3DocVQA gold identifies supporting documents but does not identify a true answer page. Safe-gate
page-hit/recovered/lost fields therefore cannot be reported for this dataset; the runner reports
document recall and downstream VQA must establish whether promoted pages are useful.

Optional diagnostic only: evaluate the `ImageListQ` subset under a page-0 proxy assumption with
`scripts/evaluate_first_page_gold_retrieval.py --baseline-pred ... --pred ... --question-type
ImageListQ --first-page-idx 0`. Report these as `synthetic_page_*` metrics and never merge them
with the exact-page results in Table C.

The structured Markdown comparison has now been run with pinned `pymupdf4llm==0.3.4`, without
OCR or automatic ONNX Layout initialization. `PyMuPDF4LLM` extracts fewer heading-bearing pages
than the native heuristic exporter (`25,355 / 44,294` versus `30,343 / 44,294`). After rerunning
the native pipeline with the current graph inputs/code, the no-heading controls align at
`doc hit@4 = 2,346`.

| M3DocVQA source | Heading pages | Safe accepted | No-heading doc hit@4 | Full-heading doc hit@4 | Strict-heading doc hit@4 | Safe-gated doc hit@4 | Safe doc net |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native, aligned rerun | 30,343 | 972 | 2,346 | 2,342 | 2,342 | 2,346 | 0 |
| `pymupdf4llm==0.3.4` | 25,355 | 766 | 2,346 | 2,348 | 2,349 | 2,346 | 0 |

At document level, PyMuPDF4LLM headings are slightly better than the native headings for direct
graph output (`+3` versus the aligned no-heading control for strict headings, while native strict
headings are `-4`). The safe gate preserves the base document result in both runs, as intended.
This still cannot establish page rescue quality because M3DocVQA has no page labels.

The earlier native `ImageListQ` page-0 proxy result (`39` to `40` synthetic hits at `@4`) was
computed before this aligned native rerun and is stale. The PyMuPDF4LLM proxy run was negative
within its own run (`43` to `42`, `1` recovered and `2` lost), but a new native proxy run is
required before making any paired proxy comparison.

ViDoSeek provides the exact-page backend check that M3DocVQA cannot. PyMuPDF4LLM conversion
completed with `0` failed or unmatched documents, but produces fewer heading pages than native
Markdown (`3,346` versus `4,342`) and a weaker direct full-heading candidate at page hit@4
(`1,019` versus `1,033`). Its gated result ties native at `1,029` page hits with `6` recovered
and `0` lost. This is a successful robustness check for the gate, not evidence that
PyMuPDF4LLM should replace the native extraction backend.

SciEGQA provides a complementary exact-page result. PyMuPDF4LLM conversion completed with `337`
heading pages and no failed or unmatched documents. Its direct candidate ties the no-heading
control at `1,323` page hits at `@4`, but the gate accepts `2` promotions and raises the output
to `1,324` with `1` recovery and `0` losses. This is a small positive extractor-diversity
result; it does not establish PyMuPDF4LLM as a globally better heading source.

## Table E: Dataset Run Status

| Dataset | Prepared? | `plain_top224` | SPLADE text source | Graph-PPR page-labeled result | Next needed action |
| --- | --- | --- | --- | --- | --- |
| M3DocVQA/MMQA | yes | yes | exported MMQA page text | aligned native/PyMuPDF4LLM document-only comparison complete; no true page labels | use downstream VQA for promotion utility; use ViDoSeek/SciEGQA for exact-page backend testing |
| MMDocIR | yes | yes | manifest/PDF text | yes | none |
| SciEGQA-Bench | yes | yes | PDF text | native and PyMuPDF4LLM exact-page safe-gate results complete | PyMuPDF4LLM yields a small guarded `+1` with zero loss |
| ViDoRe V3 | yes | yes | manifest/PDF text | yes | none for heading gate; text-derived Markdown has zero headings, so current gate is a recorded no-op |
| ViDoSeek | yes | yes | PDF text | native and PyMuPDF4LLM exact-page safe-gate results complete | retain native Markdown as primary; PyMuPDF4LLM ties gated output with weaker direct headings |
| OpenDocVQA | yes | yes | OCR-backed page text | yes | keep Graph-PPR as full-dev result; unconditional `pairwise_content_posterior` was negative |
| MMLongBench DocQA | yes | no | `page_text_list` in manifest | no | wait for embeddings, then run dense retrieval, `plain_top224`, SPLADE, and Graph-PPR |
| DUDE | yes | yes | DUDE OCR in manifest | yes | none |

## Historical Baseline Note

The old true full-dev `ret4` baseline artifact is lower than the current controlled baseline:

- old historical `rag_dev_ret4` baseline:
  - `EM = 32.4048`
  - `F1 = 37.4416`
  - `recall@4 = 0.7120`
- current controlled external baseline top4:
  - `EM = 34.7399`
  - `F1 = 40.1487`
  - `recall@4 = 0.7156`

Reason:

- old baseline = retrieve `4` directly
- current controlled baseline = retrieve `1000` with the later stronger saved retrieval artifact, then truncate to top `4` for QA

So the current controlled baseline is better, and should not be confused with the older April 10 historical baseline.

## Canonical MMQA Artifacts

### Gold

- gold JSONL:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl`

### Current Baseline

- retrieval prediction JSON:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output/retrieval_only_dev_ret1000full_nprobe4/colpali-v1.2_ivfflat_nprobe4_ret1000_2026-05-10_10-28-25.json`
- external-QA prediction JSON:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_external_qa_mmqa_dev/mmqa_dev_m3docrag_baseline_qwen2vl_top4.prediction.json`
- external-QA eval JSON:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_external_qa_mmqa_dev/mmqa_dev_m3docrag_baseline_qwen2vl_top4.eval.json`

### Historical Baseline

- old full-dev prediction JSON:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output/rag_dev_ret4/colpali-v1.2_ivfflat_ret4_Qwen2-VL-7B-Instruct_2026-04-10_22-49-57.json`
- old full-dev eval JSON:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output/rag_dev_ret4/colpali-v1.2_ivfflat_ret4_Qwen2-VL-7B-Instruct_2026-04-10_22-49-57_eval_results.json`

### MaxSim+

- `plain_top224` prediction JSON:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json`
- external-QA prediction JSON:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_external_qa_mmqa_dev/mmqa_dev_plain_top224_qwen2vl_top4.prediction.json`
- external-QA eval JSON:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_external_qa_mmqa_dev/mmqa_dev_plain_top224_qwen2vl_top4.eval.json`

### Graph Page Preserve

- graph retrieval prediction JSON:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_graph_pagepreserve_mmqa_dev/mmqa_dev_plain_top224_splade_graph_pagepreserve_denseheavy_lightboth.prediction.json`
- graph retrieval summary JSON:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_graph_pagepreserve_mmqa_dev/mmqa_dev_plain_top224_splade_graph_pagepreserve_denseheavy_lightboth.summary.json`
- graph retrieval analysis JSON:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_graph_pagepreserve_mmqa_dev/mmqa_dev_plain_top224_splade_graph_pagepreserve_denseheavy_lightboth.retrieval_analysis.json`
- graph external-QA prediction JSON:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_graph_pagepreserve_mmqa_dev/mmqa_dev_plain_top224_splade_graph_pagepreserve_denseheavy_lightboth_qwen2vl_top4.prediction.json`
- graph external-QA eval JSON:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_graph_pagepreserve_mmqa_dev/mmqa_dev_plain_top224_splade_graph_pagepreserve_denseheavy_lightboth_qwen2vl_top4.eval.json`

### Graph Page Preserve (`denseheavy125_medium_both`)

- graph retrieval prediction JSON:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_graph_pagepreserve_mmqa_dev/mmqa_dev_plain_top224_splade_graph_pagepreserve_denseheavy125_medium_both.prediction.json`
- graph retrieval summary JSON:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_graph_pagepreserve_mmqa_dev/mmqa_dev_plain_top224_splade_graph_pagepreserve_denseheavy125_medium_both.summary.json`
- graph external-QA prediction JSON:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_graph_pagepreserve_mmqa_dev/mmqa_dev_plain_top224_splade_graph_pagepreserve_denseheavy125_medium_both_qwen2vl_top4.prediction.json`
- graph external-QA eval JSON:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_graph_pagepreserve_mmqa_dev/mmqa_dev_plain_top224_splade_graph_pagepreserve_denseheavy125_medium_both_qwen2vl_top4.eval.json`

## Configs Used For These Tables

### Baseline

- retrieval artifact:
  - raw saved baseline retrieval from `ret1000full_nprobe4`
- QA:
  - external adapter
  - `qa_top_pages = 4`
  - `model_name_or_path = Qwen2-VL-7B-Instruct`
  - `bits = 16`

### MaxSim+ (`plain_top224`)

- dense source:
  - current baseline retrieval pool with `top1000`
- rerank:
  - `base-score-source = approx_page_maxsim_topk`
  - `approx-base-page-token-topk = 224`
  - `approx-base-page-token-scorer = query_mean`
  - `approx-base-page-token-selector = global_topk`
  - `approx-base-page-token-coarse-dtype = fp32`
- QA:
  - external adapter
  - `qa_top_pages = 4`
  - `model_name_or_path = Qwen2-VL-7B-Instruct`
  - `bits = 16`

### Graph Page Preserve (`denseheavy_lightboth`)

- dense input:
  - `plain_top224`
- sparse input:
  - SPLADE retrieval top1000
- graph profile:
  - `GRAPH_PROFILE = denseheavy_lightboth`
  - `dense_weight = 1.25`
  - `sparse_weight = 0.75`
  - `rrf_k = 10`
  - `doc_seed_weight = 0.0`
  - `restart_prob = 0.15`
  - `ppr_iters = 30`
  - `page_doc_edge_weight = 1.0`
  - `same_doc_window = 1`
  - `adjacent_page_edge_weight = 0.25`
  - `final_top_pages = 1000`
  - `per_doc_page_limit = 0`
  - `final_page_seed_weight = 1.0`
  - `final_ppr_page_weight = 0.25`
  - `final_ppr_doc_weight = 0.25`
- QA:
  - external adapter
  - `qa_top_pages = 4`
  - `model_name_or_path = Qwen2-VL-7B-Instruct`
  - `bits = 16`

Important:

- current wrappers default to `GRAPH_PROFILE = denseheavy125_medium_both`
- the tables above are for the earlier **`denseheavy_lightboth`** run
- keep that distinction explicit when adding future rows

### Graph Page Preserve (`denseheavy125_medium_both`)

- dense input:
  - `plain_top224`
- sparse input:
  - SPLADE retrieval top1000
- graph profile:
  - `GRAPH_PROFILE = denseheavy125_medium_both`
  - `dense_weight = 1.25`
  - `sparse_weight = 0.75`
  - `rrf_k = 10`
  - `doc_seed_weight = 0.0`
  - `restart_prob = 0.15`
  - `ppr_iters = 30`
  - `page_doc_edge_weight = 1.0`
  - `same_doc_window = 1`
  - `adjacent_page_edge_weight = 0.25`
  - `final_top_pages = 1000`
  - `per_doc_page_limit = 0`
  - `final_page_seed_weight = 1.0`
  - `final_ppr_page_weight = 0.5`
  - `final_ppr_doc_weight = 0.25`
- QA:
  - external adapter
  - `qa_top_pages = 4`
  - `model_name_or_path = Qwen2-VL-7B-Instruct`
  - `bits = 16`

## End-to-End QA Leaderboard (Top-4 Pages)

These numbers come from the final QA eval JSONs built with `qa_top_pages = 4`. The printed `recall@4` here is the baseline evaluator's average recall over the truncated top-4 page rows, so do not use these files for `recall@20`.

| Method | EM | F1 | recall@4 |
| --- | ---: | ---: | ---: |
| historical `rag_dev_ret4` baseline | `32.40` | `37.44` | `71.20%` |
| current controlled `baseline` top4 | `34.74` | `40.15` | `71.56%` |
| `MaxSim+` (`plain_top224`) | `35.64` | `41.28` | `74.89%` |
| `Graph Page Preserve` (`denseheavy_lightboth`) | `37.61` | `43.50` | `76.18%` |
| `Graph Page Preserve` (`denseheavy125_medium_both`) | `37.65` | `43.57` | `75.80%` |

Current reading:

- best end-to-end QA (`EM` / `F1`) so far:
  - `denseheavy125_medium_both`
- best top-4 recall among the evaluated QA-truncated outputs so far:
  - `denseheavy_lightboth`
- the `denseheavy125_medium_both` gain over `denseheavy_lightboth` is small but positive on answer quality:
  - `EM`: `37.6075 -> 37.6485`
  - `F1`: `43.4998 -> 43.5690`
  - `recall@4`: `76.18% -> 75.80%`

## Reproduction Commands

### 1. Baseline top4 QA eval

```bash
REPO=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
GOLD=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl
OUTDIR=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_external_qa_mmqa_dev

export LOCAL_DATA_DIR=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/data
export LOCAL_MODEL_DIR=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/model

cd "${REPO}"
python scripts/run_m3docvqa_external_retrieval_qa.py \
  --prediction-json /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output/retrieval_only_dev_ret1000full_nprobe4/colpali-v1.2_ivfflat_nprobe4_ret1000_2026-05-10_10-28-25.json \
  --gold "${GOLD}" \
  --data-name m3-docvqa \
  --split dev \
  --model-name-or-path Qwen2-VL-7B-Instruct \
  --bits 16 \
  --qa-top-pages 4 \
  --doc-image-cache-size 16 \
  --save-every 25 \
  --resume \
  --run-eval \
  --output-prediction-json "${OUTDIR}/mmqa_dev_m3docrag_baseline_qwen2vl_top4.prediction.json" \
  --output-eval-json "${OUTDIR}/mmqa_dev_m3docrag_baseline_qwen2vl_top4.eval.json"
```

### 2. MaxSim+ top4 QA eval

```bash
python scripts/run_m3docvqa_external_retrieval_qa.py \
  --prediction-json /mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json \
  --gold "${GOLD}" \
  --data-name m3-docvqa \
  --split dev \
  --model-name-or-path Qwen2-VL-7B-Instruct \
  --bits 16 \
  --qa-top-pages 4 \
  --doc-image-cache-size 16 \
  --save-every 25 \
  --resume \
  --run-eval \
  --output-prediction-json "${OUTDIR}/mmqa_dev_plain_top224_qwen2vl_top4.prediction.json" \
  --output-eval-json "${OUTDIR}/mmqa_dev_plain_top224_qwen2vl_top4.eval.json"
```

### 3. Graph Page Preserve retrieval (`denseheavy_lightboth`)

```bash
DENSE_PRED=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json \
SPARSE_PRED=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_splade_mmqa_dev/mmqa_dev_splade.prediction.json \
GRAPH_OUT_DIR=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_graph_pagepreserve_mmqa_dev \
GRAPH_PROFILE=denseheavy_lightboth \
bash scripts/run_m3docvqa_page_preserving_graph_pipeline.sh
```

### 4. Graph Page Preserve top4 QA eval (`denseheavy_lightboth`)

```bash
GRAPH_OUT_DIR=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_graph_pagepreserve_mmqa_dev \
GRAPH_PROFILE=denseheavy_lightboth \
QA_TOP_PAGES=4 \
MODEL_NAME_OR_PATH=Qwen2-VL-7B-Instruct \
bash scripts/run_m3docvqa_graph_pagepreserve_qa.sh
```

### 5. Rebuild Table A (doc hit)

```bash
python - <<'PY'
import json

gold_path = "/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl"
preds = {
    "baseline": "/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output/retrieval_only_dev_ret1000full_nprobe4/colpali-v1.2_ivfflat_nprobe4_ret1000_2026-05-10_10-28-25.json",
    "MaxSim+": "/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json",
    "Graph Page Preserve": "/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_graph_pagepreserve_mmqa_dev/mmqa_dev_plain_top224_splade_graph_pagepreserve_denseheavy_lightboth.prediction.json",
}

gold_rows = [json.loads(line) for line in open(gold_path)]
slices = {
    "Whole Dev": gold_rows,
    "ImageListQ": [row for row in gold_rows if row.get("metadata", {}).get("type") == "ImageListQ"],
}
payloads = {name: json.load(open(path)) for name, path in preds.items()}

def first_gold_rank(pred_row, gold_doc_ids):
    seen = set()
    rank = 0
    for doc_id, page_idx, score in pred_row.get("page_retrieval_results", []):
        doc_id = str(doc_id).strip()
        if doc_id in seen:
            continue
        seen.add(doc_id)
        rank += 1
        if doc_id in gold_doc_ids:
            return rank
    return None

print("| Method | Whole Dev doc@4 | Whole Dev doc@20 | ImageListQ doc@4 | ImageListQ doc@20 |")
print("| --- | ---: | ---: | ---: | ---: |")
for name, payload in payloads.items():
    vals = {}
    for slice_name, rows in slices.items():
        hit4 = 0
        hit20 = 0
        for row in rows:
            gold_doc_ids = {
                str(ctx["doc_id"]).strip()
                for ctx in row.get("supporting_context", [])
                if str(ctx.get("doc_id", "")).strip()
            }
            rank = first_gold_rank(payload[row["qid"]], gold_doc_ids)
            if rank is not None and rank <= 4:
                hit4 += 1
            if rank is not None and rank <= 20:
                hit20 += 1
        vals[(slice_name, 4)] = (hit4, len(rows), hit4 / len(rows))
        vals[(slice_name, 20)] = (hit20, len(rows), hit20 / len(rows))
    print(
        f"| {name} | "
        f"`{vals[('Whole Dev', 4)][0]} / {vals[('Whole Dev', 4)][1]} = {vals[('Whole Dev', 4)][2]*100:.2f}%` | "
        f"`{vals[('Whole Dev', 20)][0]} / {vals[('Whole Dev', 20)][1]} = {vals[('Whole Dev', 20)][2]*100:.2f}%` | "
        f"`{vals[('ImageListQ', 4)][0]} / {vals[('ImageListQ', 4)][1]} = {vals[('ImageListQ', 4)][2]*100:.2f}%` | "
        f"`{vals[('ImageListQ', 20)][0]} / {vals[('ImageListQ', 20)][1]} = {vals[('ImageListQ', 20)][2]*100:.2f}%` |"
    )
PY
```

### 6. Rebuild Table B (average recall)

```bash
python - <<'PY'
import json

gold_path = "/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl"
preds = {
    "baseline": "/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output/retrieval_only_dev_ret1000full_nprobe4/colpali-v1.2_ivfflat_nprobe4_ret1000_2026-05-10_10-28-25.json",
    "MaxSim+": "/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json",
    "Graph Page Preserve": "/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_graph_pagepreserve_mmqa_dev/mmqa_dev_plain_top224_splade_graph_pagepreserve_denseheavy_lightboth.prediction.json",
}

gold_rows = [json.loads(line) for line in open(gold_path)]
slices = {
    "Whole Dev": gold_rows,
    "ImageListQ": [row for row in gold_rows if row.get("metadata", {}).get("type") == "ImageListQ"],
}

def recall_at_k(pred_row, gold_doc_ids, k):
    rows = pred_row.get("page_retrieval_results", [])[:k]
    top_k_doc_ids = {str(row[0]).strip() for row in rows if isinstance(row, list) and len(row) >= 1}
    return len(top_k_doc_ids & gold_doc_ids) / len(gold_doc_ids) if gold_doc_ids else 0.0

payloads = {name: json.load(open(path)) for name, path in preds.items()}

print("| Method | Whole Dev recall@4 | Whole Dev recall@20 | ImageListQ recall@4 | ImageListQ recall@20 |")
print("| --- | ---: | ---: | ---: | ---: |")

for name, payload in payloads.items():
    vals = {}
    for slice_name, rows in slices.items():
        r4 = []
        r20 = []
        for row in rows:
            qid = row["qid"]
            gold_doc_ids = {
                str(ctx["doc_id"]).strip()
                for ctx in row.get("supporting_context", [])
                if str(ctx.get("doc_id", "")).strip()
            }
            pred_row = payload[qid]
            r4.append(recall_at_k(pred_row, gold_doc_ids, 4))
            r20.append(recall_at_k(pred_row, gold_doc_ids, 20))
        vals[(slice_name, 4)] = sum(r4) / len(r4)
        vals[(slice_name, 20)] = sum(r20) / len(r20)

    print(
        f"| {name} | "
        f"`{vals[('Whole Dev', 4)]*100:.2f}%` | "
        f"`{vals[('Whole Dev', 20)]*100:.2f}%` | "
        f"`{vals[('ImageListQ', 4)]*100:.2f}%` | "
        f"`{vals[('ImageListQ', 20)]*100:.2f}%` |"
    )
PY
```

## Output Structure To Reuse For Other Datasets

For each dataset, keep the same sections:

1. method labels
2. doc hit table
3. recall table
4. canonical artifact paths
5. configs used
6. reproduction commands
7. historical-baseline note if needed

Recommended artifact naming pattern:

```text
<dataset>/
  baseline/
    <dataset>_baseline.prediction.json
    <dataset>_baseline.eval.json
  maxsim_plus/
    <dataset>_plain_top224.prediction.json
    <dataset>_plain_top224.eval.json
  graph_pagepreserve/
    <dataset>_plain_top224_splade_graph_pagepreserve_<profile>.prediction.json
    <dataset>_plain_top224_splade_graph_pagepreserve_<profile>.summary.json
    <dataset>_plain_top224_splade_graph_pagepreserve_<profile>.retrieval_analysis.json
    <dataset>_plain_top224_splade_graph_pagepreserve_<profile>.eval.json
  tables/
    <dataset>_doc_hit_table.md
    <dataset>_recall_table.md
```

For MMQA, the current active profile to test next is:

- `denseheavy125_medium_both`

But keep the older `denseheavy_lightboth` rows above intact, since they are the current finalized numbers for this scoreboard.
