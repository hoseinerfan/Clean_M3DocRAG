# Retriever-Induced Graph Track

This track replaces heuristic page-promotion gates with a small label-free graph method.

## Claim

Page-preserving graph PPR can still fail page localization because document/page and adjacent-page edges mostly redistribute mass inside already retrieved documents. A retriever-induced page graph adds semantic page-page connectivity from the retrieval model itself. Pages are then ranked by PPR on independent graph views and combined with unweighted reciprocal-rank fusion (RRF).

## Kept In Main Method

1. Dense/SPLADE page retrieval seeds.
2. Base page-preserving graph PPR.
3. SPLADE-induced page kNN graph PPR.
4. Unweighted page-level RRF over graph-view rankings.

## Pruned From Main Method

1. Query-local softmax gates and boundary thresholds.
2. Threshold selectors such as `boundary_weight >= ...`.
3. Gold-informed feature audits as decision rules.
4. Constraint bundles unless they are reported as an ablation and are active on the target dataset.

## Why This Is Non-Heuristic

The method does not use gold labels, learned thresholds, or dataset-specific case rules. The SPLADE-kNN edges are generated from lexical retriever representations, PPR is a fixed graph propagation algorithm, and RRF is a standard rank aggregation method.

## Main Script

```bash
bash scripts/run_retriever_induced_graph_track.sh
```

Required environment:

```bash
DATA_NAME=vidore-v3                 # or mmdocir, opendocvqa
DATA_ROOT=/path/to/converted/root
DENSE_PRED=/path/to/plain_top224_ret1000_prediction.json
SPARSE_PRED=/path/to/splade_ret1000.prediction.json
SPLADE_INDEX_PT=/path/to/splade_page_index.pt
OUT_DIR=/path/to/graph_ppr_output
SUBSET_GOLD=/path/to/subset_or_full_gold.jsonl
SUBSET_LABEL=rankable_rightdoc_wrongpage_100
BASE_PRED="$DENSE_PRED"
export DATA_NAME DATA_ROOT DENSE_PRED SPARSE_PRED SPLADE_INDEX_PT OUT_DIR SUBSET_GOLD SUBSET_LABEL BASE_PRED
```

Outputs:

1. `${DATA_NAME}_${SUBSET_LABEL}_graph_ppr_base.prediction.json`
2. `${DATA_NAME}_${SUBSET_LABEL}_graph_ppr_splade_knn.prediction.json`
3. `${DATA_NAME}_${SUBSET_LABEL}_graph_view_rrf.prediction.json`
4. Matching summaries and top-4 case-comparison JSON files.

## Reporting Table

For each dataset/subset, report:

1. Plain dense baseline.
2. Base graph PPR.
3. SPLADE-kNN graph PPR.
4. Graph-view RRF.

Metrics:

1. Page R@1, R@2, R@4, R@10, R@20.
2. Doc R@4.
3. Recovered/lost/worsened top-4 pages versus dense baseline.
4. SPLADE-kNN edge count and source/target page counts.

## Interpretation

If SPLADE-kNN graph PPR improves top-4 page localization, the thesis claim is that semantic page-page edges address a limitation of page-preserving graph PPR. If SPLADE-kNN improves only R@1/R@2, report it as rank sharpening. If SPLADE-kNN is noisy, the unweighted RRF result is the robust final method.

## Mutual-kNN Variant

A stricter graph can be built with:

```bash
SPLADE_KNN_MUTUAL_ONLY=1 bash scripts/run_retriever_induced_graph_track.sh
```

This keeps only reciprocal SPLADE neighbors: page A connects to page B only when each appears in the other's top-k list. Mutual-kNN is a standard graph construction, not a learned or gold-tuned threshold. Use it as the first noise-control ablation when the directed SPLADE-kNN graph recovers hard cases but also worsens some same-document sibling pages.

## Layout/Evidence Graph Pivot

Full-dev SPLADE-kNN results show that semantic page-page propagation repairs some hard cases but is not a complete exact-page solution. The next limitation is page granularity: page-preserving PPR often finds the right document, then confuses same-document sibling pages. The layout/evidence graph addresses this by scoring evidence regions inside candidate pages before aggregating back to pages.

Main script:

```bash
bash scripts/run_layout_evidence_graph_track.sh
```

Required environment:

```bash
DATA_NAME=vidore-v3
DATA_ROOT=/path/to/converted/root
OUT_DIR=/path/to/graph_ppr_output
SUBSET_LABEL=rankable_rightdoc_wrongpage_100
GOLD=/path/to/subset_or_full_gold.jsonl
PREDICTION=/path/to/base_graph_or_dense_prediction.json
export DATA_NAME DATA_ROOT OUT_DIR SUBSET_LABEL GOLD PREDICTION
```

The default `PREDICTION` is `${OUT_DIR}/${DATA_NAME}_${SUBSET_LABEL}_graph_ppr_base.prediction.json` when that file exists, then `BASE_PRED`, then `DENSE_PRED`.

Method:

1. Candidate pages come from the top retrieved documents and the input page ranking.
2. Region nodes are read from corpus-side fields such as `layout_regions`, `ocr_blocks`, `text_blocks`, `tables`, `figures`, and `captions`.
3. If explicit region fields are absent, the script falls back to converted `markdown`/OCR text blocks. Report `mean_explicit_region_pages` and `mean_fallback_region_pages` so the thesis can separate true layout evidence from text-only fallback.
4. Query-region edges use BM25.
5. Region-page edges use containment.
6. Region-region edges use reading order within the page.
7. PPR produces an evidence-page ranking.
8. The final page ranking is unweighted RRF between the input ranking and the evidence graph ranking.

Why this is more defensible than another reranker:

1. The graph has typed internal evidence nodes instead of treating each page as an atomic item.
2. All edges come from corpus structure, reading order, and standard IR scoring.
3. No gold labels, learned thresholds, or dataset-specific selectors are used.
4. The summary explicitly reports whether real region/layout fields were used; text-only fallback should be framed as an immediate prototype, not the final layout claim.

### OCR Region Extraction

If converted `doc_pages_dev.jsonl` has page-level OCR text but no region fields, export OCR line/block regions for only the candidate pages used by the hard-subset experiment:

```bash
python scripts/export_ocr_region_blocks.py \
  --doc-pages-jsonl "$DATA_ROOT/doc_pages_dev.jsonl" \
  --prediction-json "$PREDICTION" \
  --qid-filter-jsonl "$GOLD" \
  --prediction-top-pages 50 \
  --ocr-engine easyocr \
  --ocr-lang en \
  --easyocr-gpu \
  --output-jsonl "$OUT_DIR/${DATA_NAME}_${SUBSET_LABEL}_ocr_regions_top50.jsonl" \
  --output-summary-json "$OUT_DIR/${DATA_NAME}_${SUBSET_LABEL}_ocr_regions_top50.summary.json" \
  --continue-on-error
```

Then rerun the evidence graph with:

```bash
REGION_JSONL="$OUT_DIR/${DATA_NAME}_${SUBSET_LABEL}_ocr_regions_top50.jsonl" \
LAYOUT_DISABLE_FALLBACK_REGIONS=1 \
LAYOUT_LABEL="${DATA_NAME}_${SUBSET_LABEL}_ocr_region_evidence_graph_top50" \
LAYOUT_CANDIDATE_SCOPE=prediction_top_pages \
LAYOUT_CANDIDATE_TOP_PAGES=50 \
OVERWRITE_LAYOUT_EVIDENCE=1 \
bash scripts/run_layout_evidence_graph_track.sh
```

With `LAYOUT_DISABLE_FALLBACK_REGIONS=1`, pages without explicit OCR/layout regions do not receive fallback text-block nodes. This makes `mean_explicit_region_pages` and `mean_fallback_region_pages` a direct sanity check for whether the result is a true OCR-region graph.

### Risk-Calibrated Evidence Gate

OCR-region graph reranking is intentionally treated as a selective operation, not an unconditional replacement for the base graph. Use `scripts/analyze_layout_evidence_gate.py` to learn a simple interpretable gate from finished base/candidate predictions and the per-query case JSON emitted by `run_layout_evidence_graph_track.sh`.

Example cross-dataset command:

```bash
python scripts/analyze_layout_evidence_gate.py \
  --run vidore "$VIDORE_GOLD" "$VIDORE_BASE" "$VIDORE_OCR_DOCANCHORED" "$VIDORE_OCR_DOCANCHORED_CASES" \
  --run mmdocir "$MMDOCIR_GOLD" "$MMDOCIR_BASE" "$MMDOCIR_OCR_DOCANCHORED" "$MMDOCIR_OCR_DOCANCHORED_CASES" \
  --hit-k 4 \
  --min-accept 5 \
  --max-doc-hit-loss 0 \
  --output-md "$OUT_DIR/layout_evidence_gate_analysis.md" \
  --output-json "$OUT_DIR/layout_evidence_gate_analysis.json" \
  --output-csv "$OUT_DIR/layout_evidence_gate_features.csv" \
  --output-rule-json "$OUT_DIR/layout_evidence_gate_rule.json" \
  --output-gated-dir "$OUT_DIR/gated_predictions"
```

The script searches single-threshold and two-condition conjunction rules using only query-time observable features:

1. Document preservation: candidate top-doc overlap with the base top documents.
2. Evidence density: positive evidence pages and positive OCR/query-region counts.
3. Rank displacement: how many candidate top pages were promoted from below the base top-k.
4. Score margins from the produced rankings.
5. Query cues such as numeric/page/visual wording.

The default rule filter enforces `--max-doc-hit-loss 0`, so the selected gate must preserve base document-hit count while improving page localization when possible. This supports a thesis claim of risk-calibrated selective OCR evidence rather than globally applying an unstable reranker.

### Adaptive Evidence Router

The stronger follow-up is a multi-candidate router, not an OCR-only gate. In this setup, `base`
means the best graph page-preserving prediction available for the dataset, preferably the
cross-dataset `denseheavy125_medium_both` graph page-preserve run. Candidate methods can include
query-anchor evidence, OCR-region evidence, structural metadata, hyperlink graphs, or any other
finished prediction JSON.

Main script:

```bash
python scripts/analyze_adaptive_evidence_router.py \
  --run mmdocir "$MMDOCIR_GOLD" "$MMDOCIR_BASE" \
  --candidate mmdocir query_anchor "$MMDOCIR_QUERY_ANCHOR" - \
  --candidate mmdocir ocr_docanchored "$MMDOCIR_OCR_DOCANCHORED" "$MMDOCIR_OCR_DOCANCHORED_CASES" \
  --run vidore "$VIDORE_GOLD" "$VIDORE_BASE" \
  --candidate vidore ocr_docanchored "$VIDORE_OCR_DOCANCHORED" "$VIDORE_OCR_DOCANCHORED_CASES" \
  --hit-k 4 \
  --min-accept 5 \
  --max-doc-hit-loss 0 \
  --max-page-hit-loss 7 \
  --max-router-rules 3 \
  --output-md "$ROUTER_DIR/adaptive_evidence_router.md" \
  --output-json "$ROUTER_DIR/adaptive_evidence_router.json" \
  --output-csv "$ROUTER_DIR/adaptive_evidence_router_features.csv" \
  --output-router-json "$ROUTER_DIR/adaptive_evidence_router_rule.json" \
  --output-routed-dir "$ROUTER_DIR/routed_predictions"
```

Use `-` as the case JSON for candidates that do not emit layout/evidence cases, such as
query-anchor graph runs. The router still has observable features from the base/candidate
rankings: page/doc overlap, rank displacement, score margins, and multilingual query cues for
quantity, visual/table, and page-locator questions.

The learned policy is an ordered list of interpretable candidate-specific rules:

```text
choose query_anchor if <query/rank/evidence condition>
else choose ocr_docanchored if <query/rank/evidence condition>
else keep base
```

Selection is greedy under `--max-doc-hit-loss 0`. This keeps the thesis claim conservative:
adaptive evidence modules may repair right-document/wrong-page failures, but the router must not
trade away document localization relative to the graph page-preserving base.

For stricter risk control, add `--max-page-hit-loss 0` to require a no-page-loss router, or set a
small budget such as `--max-page-hit-loss 7` when the goal is to improve net page hits while
limiting regressions.

Training-free self-normalized comparison:

```bash
python scripts/analyze_adaptive_evidence_router.py \
  --router-mode self_calibrated \
  --self-calibrated-profile all \
  --run mmdocir "$MMDOCIR_GOLD" "$MMDOCIR_BASE" \
  --candidate mmdocir ocr_docanchored "$MMDOCIR_OCR_DOCANCHORED" "$MMDOCIR_OCR_DOCANCHORED_CASES" \
  --run vidore "$VIDORE_GOLD" "$VIDORE_BASE" \
  --candidate vidore ocr_docanchored "$VIDORE_OCR_DOCANCHORED" "$VIDORE_OCR_DOCANCHORED_CASES" \
  --hit-k 4 \
  --max-doc-hit-loss 0 \
  --max-page-hit-loss 0 \
  --output-md "$ROUTER_DIR/adaptive_evidence_router_self_calibrated_all.md" \
  --output-json "$ROUTER_DIR/adaptive_evidence_router_self_calibrated_all.json" \
  --output-router-json "$ROUTER_DIR/adaptive_evidence_router_self_calibrated_all_rule.json" \
  --output-routed-dir "$ROUTER_DIR/routed_predictions_self_calibrated_all"
```

This mode does not learn dataset thresholds. It compares fixed, self-normalized evidence tests:

1. `robust_z`: candidate top-4 evidence must be high relative to the query's own evidence-score median and MAD.
2. `robust_z_graph_adaptive_consensus`: `robust_z` plus non-OCR support; at least one OCR-promoted top-4 page must appear by `ceil(0.20 * candidate_page_count)` in the support ranking.
3. `robust_z_graph_hit_consensus`: same as above, but the support ranking must confirm the promoted page by top-4.
4. `robust_z_graph_top10_consensus`: same as above, but the support ranking can confirm the promoted page by top-10.
5. `robust_z_graph_consensus`: same as above, but the support ranking can confirm the promoted page by top-20.
6. `robust_z_evidence_gain`: `robust_z` plus a base-vs-candidate evidence dominance test; the candidate top-4 must have stronger query-local OCR evidence than the base top-4.
7. `robust_z_qpp_veto`: `robust_z` plus a fixed margin veto requiring the base to be locally uncertain and the candidate top-4 boundary not to collapse relative to either the base boundary or the candidate head margin.
8. `percentile`: candidate top-4 evidence must land in the query's own upper evidence percentile.
9. `consensus`: the candidate must preserve base top documents while OCR evidence supports the promoted pages.
10. `pareto`: the candidate must add OCR evidence while keeping base document agreement and at least some base top-4 page support.
11. `qpp`: the base ranking must look locally uncertain and the candidate must be at least as committed at the top-4 boundary.

For `robust_z` and `percentile`, regenerate the OCR evidence graph case JSON after this code change;
OCR extraction itself does not need to be rerun.

## Boundary Reasoning And Content Evidence Update

The later right-document/wrong-page experiments moved from OCR/layout routing to a label-free
local boundary reasoner. The problem setting is top-4 page localization when the gold page is
near the decision boundary, usually rank 5-10, while the document is already correct.

Subset baseline on `rankable_rightdoc_wrongpage_100`:

| dataset | base page hit@4 | base doc hit@4 |
| --- | ---: | ---: |
| ViDoRe V3 | 0 | 100 |
| MMDocIR | 45 | 97 |

Method audit:

| category | method | ViDoRe recovered/lost/net | MMDocIR recovered/lost/net | MMDocIR page hit@4 | conclusion |
| --- | --- | ---: | ---: | ---: | --- |
| OCR/layout evidence | raw `ocr_docanchored` | 49 / 0 / +49 | about 13 / 14-17 / negative-to-flat | about 42-44 | strong on ViDoRe, unstable on MMDocIR |
| self-calibrated OCR | `robust_z` | 34 / 0 / +34 | 13 / 14 / -1 | 44 | still loses too many MMDocIR hits |
| graph support gate | `robust_z_graph_top10_consensus` | 34 / 0 / +34 | 13 / 13 / 0 | 45 | safer, limited upside |
| boundary Gaussian | `relative_z` | 58 / 0 / +58 | 10 / 5 / +5 | 50 | first useful non-OCR boundary rescue |
| paired Gaussian | `paired_gaussian c90` | 40 / 0 / +40 | 1 / 1 / 0 | 45 | too conservative |
| full posterior | `posterior_rerank` | 85 / 0 / +85 | 20 / 13 / +7 | 52 | large ViDoRe gain, too lossy on MMDocIR |
| pairwise posterior | `pairwise_posterior` | 64 / 0 / +64 | 7 / 2 / +5 | 50 | best risk-adjusted method before content |
| preserve prior | `pairwise_posterior_preserve` | 48 / 0 / +48 | 3 / 1 / +2 | 47 | too conservative |
| adaptive preserve | `pairwise_posterior_adaptive_preserve` | 49 / 0 / +49 | 4 / 1 / +3 | 48 | still over-preserves base top-4 |
| counterfactual posterior | `pairwise_counterfactual_posterior` | 48 / 0 / +48 | 3 / 1 / +2 | 47 | base-only null model dominates |
| evidence-only counterfactual | `pairwise_counterfactual_evidence_posterior` | 57 / 0 / +57 | 6 / 3 / +3 | 48 | useful diagnostic, not best |
| content posterior | `pairwise_content_posterior` | 62 / 0 / +62 | 13 / 3 / +10 | 55 | best risk-adjusted subset result so far |
| content counterfactual | `pairwise_counterfactual_content_posterior` | 58 / 0 / +58 | 12 / 2 / +10 | 55 | safer ablation, same MMDocIR net |
| exact MaxSim boundary | `rerank_graph_boundary_exact_maxsim.py` | pending | pending | pending | label-free top-4/rank-5 verifier for full Graph-PPR |

Routing/selection audits:

| router | ViDoRe net | MMDocIR recovered/lost/net | MMDocIR page hit@4 | note |
| --- | ---: | ---: | ---: | --- |
| query subtype LORO, non-oracle | +58 | 20 / 11 / +9 | 54 | useful but still loses too many top-4 pages |
| agreement signature raw | +56 | 19 / 11 / +8 | 53 | rank signatures only, no case/gold leakage |
| agreement signature LCB c90 n5 | 0 | 12 / 9 / +3 | 48 | too conservative cross-dataset |
| agreement signature LCB c75 n3 | 0 | 14 / 9 / +5 | 50 | weaker than direct content posterior |

Current interpretation:

1. More graph/rank routing alone does not separate valid boundary pages from same-document distractors.
2. Direct query-page content evidence is the first independent signal that improves MMDocIR without destroying ViDoRe.
3. `pairwise_content_posterior` is the current main hard-subset boundary method; `pairwise_counterfactual_content_posterior` is a conservative ablation.
4. Do not apply `pairwise_content_posterior` unconditionally to full datasets. The full OpenDocVQA no-support run shows it is unsafe when the base already has many correct top-4 pages.

Full Graph-PPR limitation report:

The audit below is oracle analysis only. It explains remaining failures of the frozen
`denseheavy125_medium_both` Graph-PPR output and must not be used as a routing feature.

| Dataset | qids | page hit@4 | doc hit@4 | page failures | document retrieval gap | same-document page confusion | rank-boundary localization | right-doc deep/missing page |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViDoRe V3 | 14514 | 9383 | 13206 | 5131 | 1308 / 5131 = 25.5% | 1340 / 5131 = 26.1% | 2228 / 5131 = 43.4% | 255 / 5131 = 5.0% |
| MMDocIR | 1658 | 1114 | 1353 | 544 | 305 / 544 = 56.1% | 74 / 544 = 13.6% | 147 / 544 = 27.0% | 18 / 544 = 3.3% |
| OpenDocVQA | 41017 | 26173 | 26901 | 14844 | 14116 / 14844 = 95.1% | 119 / 14844 = 0.8% | 597 / 14844 = 4.0% | 12 / 14844 = 0.1% |

Primary failure category split from the latest rich run:

| Dataset | doc_miss_topk | doc_missing_from_pool | right_doc_boundary_page | right_doc_adjacent_page | right_doc_same_doc_sibling | right_doc_late_page | right_doc_gold_page_missing_from_pool |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViDoRe V3 | 1235 | 73 | 2228 | 335 | 1005 | 241 | 14 |
| MMDocIR | 248 | 57 | 147 | 15 | 59 | 15 | 3 |
| OpenDocVQA | 13552 | 564 | 597 | 5 | 114 | 11 | 1 |

Rank-5 and document-rank diagnostics:

| Dataset | rank-5 gold failures | right-doc failures | rank-5 gold within right-doc failures | dominant failed gold-doc bucket |
| --- | ---: | ---: | ---: | --- |
| ViDoRe V3 | 470 / 5131 = 9.2% | 3823 | 449 / 3823 = 11.7% | `top4` = 3823 |
| MMDocIR | 16 / 544 = 2.9% | 239 | 15 / 239 = 6.3% | `top4` = 239 |
| OpenDocVQA | 1184 / 14844 = 8.0% | 728 | 207 / 728 = 28.4% | `doc_5_10` = 5220 |

Implications:

1. ViDoRe is the strongest target for local page evidence: most failures already have the right document in the top 4, and finance/table-heavy subsets show many same-document and boundary mistakes.
2. MMDocIR needs both levels. Boundary/content evidence can recover some cases, but the largest failure group is document discovery.
3. OpenDocVQA needs document/pack selection before page rescue. The boundary-looking page ranks hide a document-rank problem: most failures have the gold document outside the top 4 or missing from the retrieved pool.
4. Exact MaxSim top4-vs-rank5 is still useful as a narrow non-OCR diagnostic, but the report limits its expected ceiling: it directly targets only rank-5/right-document cases, not deep document-retrieval gaps.

The regenerated report now also prints the categorical views needed for the limitation write-up:
retrievability ceiling, failure category by limitation group, limitation by page-rank bucket,
limitation by document-rank bucket, query-cue slices, gold-label shape, top-k evidence tags,
score-margin diagnostics, and metadata hotspots split by metadata field.

Canonical subset command shape:

```bash
python scripts/rerank_boundary_gaussian_graph.py \
  --gold "$GOLD" \
  --base-prediction "$BASE" \
  --support graph_view "$GRAPH_SUPPORT" \
  --doc-pages-jsonl "$DOC_PAGES" \
  --decision-test pairwise_content_posterior \
  --hit-k 4 \
  --boundary-top-pages 10 \
  --output-prediction-json "$BOUNDARY_DIR/${RUN_LABEL}_boundary_pairwise_content_posterior.prediction.json" \
  --output-summary-json "$BOUNDARY_DIR/${RUN_LABEL}_boundary_pairwise_content_posterior.summary.json" \
  --output-case-json "$BOUNDARY_DIR/${RUN_LABEL}_boundary_pairwise_content_posterior.cases.json"
```

When no independent support prediction exists, omit the `--support ...` line rather than passing the
base prediction as support.

Exact MaxSim boundary verifier command shape:

```bash
python scripts/rerank_graph_boundary_exact_maxsim.py \
  --gold "$GOLD" \
  --base-prediction "$BASE" \
  --embedding-dir "$EMBEDDING_DIR" \
  --hit-k 4 \
  --boundary-rank 5 \
  --output-prediction-json "$MAXSIM_BOUNDARY_DIR/${RUN_LABEL}_exact_maxsim_boundary.prediction.json" \
  --output-summary-json "$MAXSIM_BOUNDARY_DIR/${RUN_LABEL}_exact_maxsim_boundary.summary.json" \
  --output-case-json "$MAXSIM_BOUNDARY_DIR/${RUN_LABEL}_exact_maxsim_boundary.cases.json"
```

This scorer is a local verifier rather than a learned selector: it recomputes exact ColPali MaxSim
for the current top-4 pages and rank 5 only, then swaps rank 5 into top 4 if exact MaxSim beats the
weakest current top-4 page. The result is still experimental until full-dataset losses/recoveries
are known.

After the MMDocIR full-dev run was negative, use the gated subset mode before trying another full
run. The gate is still observable-only: deterministic sample, document-neighborhood policy, optional
base-margin uncertainty, and exact MaxSim margin.

```bash
MAXSIM_BOUNDARY_DIR=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/boundary_exact_maxsim
mkdir -p "$MAXSIM_BOUNDARY_DIR"

python scripts/rerank_graph_boundary_exact_maxsim.py \
  --gold "$MMDOCIR_DATA/MMQA_dev.jsonl" \
  --base-prediction "$MMDOCIR_OUT/mmdocir_dev_graph_ppr_base.prediction.json" \
  --embedding-dir "$MMDOCIR_ROOT/embeddings/colpali-v1.2_mm-docir_dev" \
  --hit-k 4 \
  --boundary-rank 5 \
  --sample-qids 300 \
  --sample-seed 17 \
  --boundary-doc-policy weakest_doc \
  --min-exact-margin 0.25 \
  --output-prediction-json "$MAXSIM_BOUNDARY_DIR/mmdocir_exact_maxsim_boundary_gated_weakestdoc_m025_sample300.prediction.json" \
  --output-summary-json "$MAXSIM_BOUNDARY_DIR/mmdocir_exact_maxsim_boundary_gated_weakestdoc_m025_sample300.summary.json" \
  --output-case-json "$MAXSIM_BOUNDARY_DIR/mmdocir_exact_maxsim_boundary_gated_weakestdoc_m025_sample300.cases.json"
```

Observed MMDocIR 300-query subset results:

| policy | sample | accepted | base page hit@4 | candidate page hit@4 | recovered | lost | net | page recall@4 | doc recall@4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `weakest_doc`, `min_exact_margin=0.25` | 300 | 43 | 186 | 184 | 0 | 2 | -2 | 0.5884 | 0.7900 |
| `topk_doc`, `min_exact_margin=0.25` | 300 | 57 | 186 | 183 | 0 | 3 | -3 | 0.5834 | 0.7900 |

Do not scale either observed gated setting as-is. The relaxed document policy accepted more swaps
but only increased losses, so exact rank-5 MaxSim is not currently a reliable MMDocIR page-evidence
verifier.

Full OpenDocVQA no-support result:

| run | qids | accepted | base page hit@4 | candidate page hit@4 | recovered | lost | net | page recall@4 | doc recall@4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `opendocvqa_boundary_pairwise_content_posterior_nosupport` | 41017 | 10209 | 26173 | 22592 | 146 | 3727 | -3581 | 0.5027 | 0.5683 |

Interpretation: this is a negative full-dev result. The method was learned from a hard subset where
base top-4 page hit was absent or weak, so losses were structurally limited. On full OpenDocVQA, base
page hit@4 is already high; unconditional boundary swaps destroy many correct base top-4 pages. Treat
content posterior as a diagnostic/rescue component until there is a non-oracle selector for when the
base top-4 page set is likely wrong.

Cross-dataset conditioning requirement:

The conditioning step must be evaluated as a transfer problem, not tuned on one dataset. Use
`scripts/learn_query_subtype_router.py` with leave-one-run-out evaluation and equal-run weighting.
This makes the selector learn from the other datasets and then test on the held-out dataset; OpenDocVQA
does not dominate training just because it has many more queries.

```bash
python scripts/learn_query_subtype_router.py \
  --run vidore "$VIDORE_GOLD" "$VIDORE_BASE" \
  --candidate vidore content "$BOUNDARY_DIR/vidore_boundary_pairwise_content_posterior.prediction.json" "$BOUNDARY_DIR/vidore_boundary_pairwise_content_posterior.cases.json" \
  --run mmdocir "$MMDOCIR_GOLD" "$MMDOCIR_BASE" \
  --candidate mmdocir content "$BOUNDARY_DIR/mmdocir_boundary_pairwise_content_posterior.prediction.json" "$BOUNDARY_DIR/mmdocir_boundary_pairwise_content_posterior.cases.json" \
  --run opendocvqa "$OPENDOC_GOLD" "$OPENDOC_BASE" \
  --candidate opendocvqa content "$BOUNDARY_DIR/opendocvqa_boundary_pairwise_content_posterior_nosupport.prediction.json" "$BOUNDARY_DIR/opendocvqa_boundary_pairwise_content_posterior_nosupport.cases.json" \
  --hit-k 4 \
  --cv-mode leave_run_out \
  --run-weighting equal_run \
  --output-json "$ROUTER_DIR/content_boundary_router_loro_equalrun.json" \
  --output-md "$ROUTER_DIR/content_boundary_router_loro_equalrun.md" \
  --output-routed-dir "$ROUTER_DIR/routed_content_boundary_loro_equalrun"
```

This is the required test before claiming a full-dataset conditioned boundary method. If the held-out
OpenDocVQA row is still negative, keep Graph-PPR as the full-dataset method and report content posterior
only as a hard-subset rescue component.

### Cluster-Conditioned Router

`scripts/learn_cluster_conditioned_router.py` implements the unsupervised query-clustering direction.
It clusters each candidate's observable query/rank/content features with deterministic weighted k-means,
selects the cluster count by a spherical-Gaussian BIC score on the training fold, and estimates page/doc
utility per cluster. A held-out query is routed to a candidate only when its assigned cluster has positive
page-hit utility and nonnegative doc-hit utility.

This is the complete cross-dataset test shape:

```bash
python scripts/learn_cluster_conditioned_router.py \
  --run vidore "$VIDORE_GOLD" "$VIDORE_BASE" \
  --candidate vidore content "$BOUNDARY_DIR/vidore_boundary_pairwise_content_posterior.prediction.json" "$BOUNDARY_DIR/vidore_boundary_pairwise_content_posterior.cases.json" \
  --run mmdocir "$MMDOCIR_GOLD" "$MMDOCIR_BASE" \
  --candidate mmdocir content "$BOUNDARY_DIR/mmdocir_boundary_pairwise_content_posterior.prediction.json" "$BOUNDARY_DIR/mmdocir_boundary_pairwise_content_posterior.cases.json" \
  --run opendocvqa "$OPENDOC_GOLD" "$OPENDOC_BASE" \
  --candidate opendocvqa content "$BOUNDARY_DIR/opendocvqa_boundary_pairwise_content_posterior_nosupport.prediction.json" "$BOUNDARY_DIR/opendocvqa_boundary_pairwise_content_posterior_nosupport.cases.json" \
  --hit-k 4 \
  --cv-mode leave_run_out \
  --run-weighting equal_run \
  --cluster-count 0 \
  --min-clusters 1 \
  --max-clusters 0 \
  --min-cluster-n 5 \
  --doc-policy nonnegative \
  --output-json "$ROUTER_DIR/cluster_conditioned_content_loro_equalrun.json" \
  --output-md "$ROUTER_DIR/cluster_conditioned_content_loro_equalrun.md" \
  --output-routed-dir "$ROUTER_DIR/routed_cluster_conditioned_content_loro_equalrun"
```

Use the same command with `pairwise_posterior` artifacts instead of `pairwise_content_posterior`
artifacts for a strictly non-content/non-OCR backup test.
