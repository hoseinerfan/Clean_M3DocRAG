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
