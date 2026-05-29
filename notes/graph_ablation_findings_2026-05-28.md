# Graph Ablation Findings 2026-05-28

Purpose: consolidate the recent Graph-PPR ablations so the main notes and generated result tables have a stable interpretation layer. The raw generated tables remain in dataset output folders; this file records the conclusions that should guide the next runs and thesis writeup.

Latest focused M3DocVQA update: [m3docvqa_gpp_ablation_findings_2026-05-29.md](/Users/hoseinerfan/Desktop/Clean_M3DocRAG/notes/m3docvqa_gpp_ablation_findings_2026-05-29.md:1)

## Executive Summary

- For MMDocIR, SciEGQA, ViDoSeek, and ViDoRe page-labeled retrieval, doc-doc edges are not a reliable page@4 improvement. OpenDocVQA is the important exception: `fully_connected_topdocs` improves average page recall@4 by `+0.0140` and average doc recall@4 by `+0.0153`.
- Doc-seed restart helps SciEGQA but not MMDocIR or ViDoSeek. The best SciEGQA doc-seed row was `docseed_rrf_1p00`, improving page@4 by `+12` and doc@4 by `+4`.
- New selected-dataset runs show OpenDocVQA doc-seed hurts, ViDoRe doc-seed is only a tiny page gain with doc-rank cost, and ViDoRe hard cross-doc page selection is harmful.
- M3DocVQA is different: it has document-only gold labels, and the authored PDF/Wikipedia hyperlink graph gives small but real doc/row gains. The latest best GPP hyperlink variant is `pagenode_to_hyperlink_pages + log_count + PDF_HYPERLINK_TARGET_PAGES_PER_DOC=1 + MMR doc-diverse`.
- The current best M3DocVQA hyperlink-node run reaches `doc@4=0.853`, `doc@20=0.936`, `row@4=0.804`, and `row@20=0.905`. It trades off top-1 recall (`doc@1=0.600` vs `0.608` no-hyperlink), so `docnode_to_hyperlink_docs` remains the conservative alternative.
- The new M3DocVQA dev-split doc-fusion probe is positive: tuned fusion reaches `eval doc@4=0.8613`, beating the best single eval source by `+0.0094`. The best weights were dense `2.0`, SPLADE `4.0`, no-hyperlink GPP `0.0`, doc-hyperlink GPP `0.25`, and page-hyperlink GPP `4.0`.
- Cross-doc/final row selection is still important for M3DocVQA: MMR document-diverse selection raises row@4 substantially compared with score-only selection, and the hyperlink-node variants build on that.
- `shared_entity_title_topic` is currently a no-op on the checked datasets because it emits zero edges.
- `semantic_similarity` was skipped in the complete M3DocVQA run only because `SPLADE_INDEX_PT` was not pointed at the existing SPLADE index. The full index exists at `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_splade/m3docvqa_dev_splade.pt`.

## Page-Labeled Gold Structure

The current page-labeled datasets do not all need cross-document page selection.

| Dataset | qids | multi-gold-page qids | multi-page same-doc only | multi-page different-docs | Implication |
|---|---:|---:|---:|---:|---|
| MMDocIR | 1,658 | 312 | 312 | 0 | Cross-doc page diversity is not needed for gold coverage. |
| SciEGQA | 1,623 | 318 | 318 | 0 | Same-doc page localization matters more than cross-doc diversity. |
| ViDoSeek | 1,142 | 0 | 0 | 0 | Single-page, saturated retrieval. |
| DUDE | 2,903 | 59 | 59 | 0 | Mostly single-page/same-doc. |
| ViDoRe V3 | 14,514 | 10,638 | 9,252 | 1,386 | Cross-doc page diversity is relevant. |
| OpenDocVQA | 41,017 | 9,412 | 64 | 9,348 | Cross-doc page diversity is highly relevant. |
| MMLongBench DocQA | 14,466 | 4,345 | 4,345 | 0 | Same-doc multi-page evidence, plus 1,050 no-page-gold qids. |
| M3DocVQA | 2,441 | 0 | 0 | 0 | Document-only gold; row/page rank is diagnostic, not page-gold recall. |

## External Doc-Doc Edge Ablation

Baseline for these deltas is the page-preserving Graph-PPR `no_doc_doc` row:

| Dataset | baseline page@4 count | baseline doc@4 count | best doc-doc outcome | Finding |
|---|---:|---:|---|---|
| MMDocIR | 1114 | 1353 | `fully_connected_topdocs` gives doc@4 `+5` but page@4 `-7` | Doc-doc edges can improve document ranking while hurting exact page ranking. |
| SciEGQA | 1323 | 1508 | `no_doc_doc` remains best for both page@4 and doc@4 | Doc-doc edges hurt this dataset. |
| ViDoSeek | 1020 | 1142 | all doc-doc variants keep doc@4 saturated and page@4 unchanged | Saturated; no meaningful gain. |
| ViDoRe V3 | 9383 | 13206 | no useful average page recall gain | Doc-doc edges are effectively neutral or slightly harmful. |
| OpenDocVQA | 26173 | 26901 | `fully_connected_topdocs` gives page hit@4 `+625` and doc hit@4 `+684` | Strong non-hyperlink gain. |

Detailed page@4/doc@4 deltas:

| Dataset | Variant | delta page@4 | delta doc@4 | edge qids | mean edge pairs |
|---|---|---:|---:|---:|---:|
| MMDocIR | dense_sparse_agreement | -3 | +3 | 1658 | 70.36 |
| MMDocIR | fully_connected_topdocs | -7 | +5 | 1658 | 190.00 |
| MMDocIR | semantic_similarity | 0 | 0 | 1375 | 15.32 |
| MMDocIR | shared_entity_title_topic | 0 | 0 | 0 | 0.00 |
| SciEGQA | dense_sparse_agreement | -3 | -2 | 1623 | 72.96 |
| SciEGQA | fully_connected_topdocs | -5 | -3 | 1623 | 190.00 |
| SciEGQA | semantic_similarity | 0 | 0 | 1039 | 5.60 |
| SciEGQA | shared_entity_title_topic | 0 | 0 | 0 | 0.00 |
| ViDoSeek | dense_sparse_agreement | 0 | 0 | 1142 | 71.74 |
| ViDoSeek | fully_connected_topdocs | 0 | 0 | 1142 | 190.00 |
| ViDoSeek | semantic_similarity | 0 | 0 | 233 | 0.34 |
| ViDoSeek | shared_entity_title_topic | 0 | 0 | 0 | 0.00 |

Audit checks passed for dense/sparse agreement and feature mechanisms:

- dense/sparse agreement had `mismatch_count=0` and `bad_pair_qids=0` on all three datasets.
- `shared_entity_title_topic` emitted zero edges on all checked datasets.
- `semantic_similarity` emitted edges and audited correctly, but did not improve page@4.

Selected OpenDocVQA/ViDoRe average recall deltas:

| Dataset | Variant | delta avg page recall@4 | delta avg doc recall@4 | delta page hit@4 | delta doc hit@4 |
|---|---|---:|---:|---:|---:|
| ViDoRe V3 | dense_sparse_agreement | -0.0004 | -0.0003 | +1 | -4 |
| ViDoRe V3 | fully_connected_topdocs | -0.0011 | +0.0000 | -11 | +2 |
| ViDoRe V3 | semantic_similarity | -0.0002 | -0.0004 | 0 | -6 |
| ViDoRe V3 | all_doc_doc_features | -0.0003 | -0.0003 | +2 | -5 |
| OpenDocVQA | dense_sparse_agreement | +0.0037 | +0.0042 | +134 | +158 |
| OpenDocVQA | fully_connected_topdocs | +0.0140 | +0.0153 | +625 | +684 |
| OpenDocVQA | semantic_similarity | +0.0002 | +0.0004 | +4 | +13 |
| OpenDocVQA | all_doc_doc_features | +0.0029 | +0.0032 | +99 | +116 |

## External Doc-Seed Ablation

Baseline is `docseed_none`.

| Dataset | baseline page@4 count | baseline doc@4 count | best row | delta page@4 | delta doc@4 | Finding |
|---|---:|---:|---|---:|---:|---|
| MMDocIR | 1114 | 1353 | no useful doc-seed row | at most +1 | negative doc@4 deltas | Do not use doc-seed here. |
| SciEGQA | 1323 | 1508 | `docseed_rrf_1p00` | +12 | +4 | Strongest external doc-seed result. |
| ViDoSeek | 1020 | 1142 | no useful doc-seed row | 0 or -1 | 0 | Saturated/no gain. |
| ViDoRe V3 | 9383 | 13206 | `docseed_rrf_1p00` gives page hit@4 `+29` but doc hit@4 `-21` | tiny positive | negative | Too weak to use as main setting. |
| OpenDocVQA | 26173 | 26901 | no useful doc-seed row | negative | negative | Do not use doc-seed here. |

SciEGQA doc-seed progression:

| Variant | page@4 count | delta page@4 | doc@4 count | delta doc@4 |
|---|---:|---:|---:|---:|
| docseed_none | 1323 | 0 | 1508 | 0 |
| docseed_rrf_0p25 | 1325 | +2 | 1510 | +2 |
| docseed_rrf_0p50 | 1328 | +5 | 1511 | +3 |
| docseed_rrf_1p00 | 1335 | +12 | 1512 | +4 |

Selected OpenDocVQA/ViDoRe doc-seed average recall deltas:

| Dataset | Variant | delta avg page recall@4 | delta avg doc recall@4 | delta page hit@4 | delta doc hit@4 |
|---|---|---:|---:|---:|---:|
| ViDoRe V3 | docseed_rrf_0p25 | +0.0004 | 0.0000 | +10 | 0 |
| ViDoRe V3 | docseed_rrf_0p50 | +0.0006 | -0.0004 | +22 | -5 |
| ViDoRe V3 | docseed_rrf_1p00 | +0.0006 | -0.0017 | +29 | -21 |
| ViDoRe V3 | docseed_avgpage_0p50 | -0.0002 | -0.0004 | -6 | -4 |
| OpenDocVQA | docseed_rrf_0p25 | -0.0018 | -0.0024 | -87 | -112 |
| OpenDocVQA | docseed_rrf_0p50 | -0.0033 | -0.0042 | -151 | -192 |
| OpenDocVQA | docseed_rrf_1p00 | -0.0050 | -0.0065 | -231 | -295 |
| OpenDocVQA | docseed_avgpage_0p50 | -0.0090 | -0.0099 | -409 | -448 |

## Selected Cross-Doc Page Selection Ablation

The ViDoRe V3 selected run was interrupted at `mmr_docdiv_pool20_b0p05`, so only completed rows should be interpreted.

| Dataset | Variant | delta avg page recall@4 | delta avg doc recall@4 | delta page hit@4 | delta doc hit@4 | Finding |
|---|---|---:|---:|---:|---:|---|
| ViDoRe V3 | `max1doc_pool20` | -0.0686 | 0.0000 | -898 | 0 | Harmful over-diversification. |
| ViDoRe V3 | `max1doc_pool50` | -0.0975 | 0.0000 | -1430 | 0 | Very harmful. |
| ViDoRe V3 | `mmr_docdiv_pool20_b0p02` | -0.0016 | 0.0000 | +34 | 0 | Hit count up, average recall down; MMR sweep incomplete. |

## M3DocVQA Latest Focused Update

Detailed note: [m3docvqa_gpp_ablation_findings_2026-05-29.md](/Users/hoseinerfan/Desktop/Clean_M3DocRAG/notes/m3docvqa_gpp_ablation_findings_2026-05-29.md:1)

Current best config:

```bash
PDF_HYPERLINK_WEIGHT_MODE=log_count
PDF_HYPERLINK_TARGET_MODE=target_pages
PDF_HYPERLINK_TARGET_PAGES_PER_DOC=1
FINAL_SELECTION_MODE=mmr_doc_diverse
FINAL_SELECTION_NEW_DOC_BONUS=0.10
FINAL_SELECTION_SAME_DOC_PENALTY=0.05
```

Latest MMR hyperlink-node rows:

| Method | doc@1 | doc@4 | doc@10 | doc@20 | row@4 | row@10 | row@20 | Main reading |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| no hyperlink MMR | 0.608 | 0.846 | 0.903 | 0.927 | 0.788 | 0.849 | 0.893 | Current no-hyperlink baseline. |
| `docnode_to_hyperlink_docs` | 0.608 | 0.851 | 0.907 | 0.930 | 0.798 | 0.855 | 0.898 | Conservative hyperlink gain; preserves top-1. |
| `pagenode_to_hyperlink_pages`, target pages 4 | 0.604 | 0.854 | 0.908 | 0.933 | 0.768 | 0.857 | 0.899 | Best doc@4, but row@4 drops because target pages are diffuse. |
| `pagenode_to_hyperlink_pages`, target pages 1 | 0.600 | 0.853 | 0.911 | 0.936 | 0.804 | 0.864 | 0.905 | Best current row/context and deeper recall. |

Dev-split doc-fusion probe:

| Method | eval doc@4 | Delta vs best source | Meaning |
|---|---:|---:|---|
| best single eval source | 0.8518 | 0.0000 | Best individual source on the eval split. |
| tuned doc fusion | 0.8613 | +0.0094 | Positive document-level tuning signal. |

Best tuned weights: dense `2.0`, SPLADE `4.0`, no-hyperlink GPP `0.0`, doc-hyperlink GPP `0.25`, page-hyperlink GPP `4.0`.

Other M3DocVQA findings from today:

- Doc-seed page-score variants do not beat the no-docseed baseline in a useful way. Sum scoring gives only tiny `doc@4` gains and hurts `row@4`.
- Adjacent-page edge weight is weak on M3DocVQA. Higher weights slightly improve some doc@4 rows, but hurt row context for multi-gold-doc qids.
- M3DocVQA train has qids and gold docs, but no gold pages. It can support doc-level tuning, not true page-level supervised tuning.
- OpenDocVQA, ViDoRe, DUDE, ViDoSeek, and SciEGQA do not yet have usable mapped dataset-internal hyperlink edges from the sanity checks. The hyperlink method is currently M3DocVQA-only.

## M3DocVQA Historical Hyperlink Graph

The M3DocVQA hyperlink file is valid and useful:

| Item | Count |
|---|---:|
| valid hyperlink edges | 21,451 |
| source pages | 13,451 |
| source docs | 2,885 |
| target docs | 2,417 |
| qids with any gold target inlink | 1,959 / 2,441 |
| qids with retrieved link to a gold doc | 1,936 / 2,441 |

The tuned hyperlink weight sweep used the same `no_doc_doc` baseline:

| weight | doc@1 | doc@4 | doc@10 | row@4 | row@10 | improved / worsened |
|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.608 | 0.846 | 0.903 | 0.758 | 0.848 | - |
| 1.75 | 0.608 | 0.851 | 0.906 | 0.769 | 0.854 | 37 / 35 |
| 2.00 | 0.608 | 0.851 | 0.907 | 0.769 | 0.854 | 38 / 37 |
| 2.25 | 0.608 | 0.851 | 0.907 | 0.770 | 0.854 | 40 / 37 |
| 2.50 | 0.608 | 0.851 | 0.907 | 0.770 | 0.855 | 40 / 39 |
| 3.00 | 0.607 | 0.851 | 0.907 | 0.770 | 0.856 | 40 / 44 |
| 5.00 | 0.605 | 0.850 | 0.907 | 0.769 | 0.856 | 43 / 59 |
| 10.00 | 0.602 | 0.849 | 0.908 | 0.770 | 0.857 | 50 / 76 |
| 20.00 | 0.603 | 0.846 | 0.907 | 0.771 | 0.857 | 55 / 84 |

Historical M3DocVQA hyperlink setting before the newer hyperlink-node target-page runs:

```bash
DOC_DOC_EDGE_MODE=hyperlink_citation
DOC_DOC_EDGE_WEIGHT=2.25
```

Rationale: `2.25` keeps doc@1 unchanged, reaches the best observed doc@4 band in that older doc-doc edge sweep, improves row@4 to `0.770`, and has the best movement balance in the `1.75-3.00` local sweep. For current GPP reporting, prefer the newer hyperlink-node target-page MMR result above.

## M3DocVQA Complete Ablation

The complete wrapper was run with the default doc-doc edge weight (`0.10`) unless the environment overrides it, so it should not replace the tuned hyperlink weight result above.

| Method | doc@4 | row@4 | Main finding |
|---|---:|---:|---|
| `docdoc_no_doc_doc` | 0.846 | 0.758 | Baseline. |
| `docdoc_dense_sparse_agreement` | 0.845 | 0.758 | Slightly worse doc@4. |
| `docdoc_fully_connected_topdocs` | 0.847 | 0.758 | Small doc@4 gain, no row@4 gain. |
| `docdoc_hyperlink_citation` default `w0.10` | 0.847 | 0.759 | Small gain; tuned `w2.25` is better. |
| `docdoc_all_doc_doc_features` | 0.847 | 0.759 | Small gain only. |
| `docseed_rrf_1p00` | 0.840 | 0.761 | Row@4 up, doc@4 down; not a main setting. |
| `select_max1doc_pool50` | 0.846 | 0.845 | Strong row diversity effect, doc@4 unchanged. |
| `select_mmr_docdiv_pool20_b0p10` | 0.846 | 0.799 | Best row@4 among complete rows, doc@4 unchanged. |

Historical combined M3DocVQA test from the older doc-doc sweep:

```bash
DOC_DOC_EDGE_WEIGHT=2.25 \
FINAL_SELECTION_MODE=mmr_doc_diverse \
FINAL_SELECTION_CANDIDATE_POOL=20 \
FINAL_SELECTION_NEW_DOC_BONUS=0.10 \
GRAPH_OUT_DIR=output/m3docvqa_hyperlink_w2p25_mmr_b0p10 \
bash examples/run_m3docvqa_doc_doc_edge_ablation.sh
```

The direct semantic-similarity rerun should use:

```bash
export SPLADE_INDEX_PT=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_splade/m3docvqa_dev_splade.pt
```

and explicit dense/sparse prediction paths if running the lower-level pipeline directly.

## Reporting Recommendation

- For MMDocIR/SciEGQA/ViDoSeek page-labeled retrieval, report `denseheavy125_medium_both` with `DOC_DOC_EDGE_MODE=none` as the main page-preserving graph result.
- For SciEGQA only, report `docseed_rrf_1p00` as a useful doc-seed ablation.
- For M3DocVQA, report the latest hyperlink-node result as the main GPP hyperlink result:
  - current best: `pagenode_to_hyperlink_pages + log_count + PDF_HYPERLINK_TARGET_PAGES_PER_DOC=1 + MMR doc-diverse`
  - conservative alternative: `docnode_to_hyperlink_docs + log_count + MMR doc-diverse`
  - older `DOC_DOC_EDGE_WEIGHT=2.25` remains useful historical comparison, but it is superseded for current GPP hyperlink reporting
- Do not claim `shared_entity_title_topic` helped; it emitted zero edges.
- Do not claim M3DocVQA semantic-similarity failed until rerunning with the found SPLADE index.
