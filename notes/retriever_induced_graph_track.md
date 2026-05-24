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
