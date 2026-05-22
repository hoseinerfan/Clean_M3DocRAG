# Graph-PPR External Dataset Handoff

Date: 2026-05-21

Purpose: use this note in the datasets chat to adjust Graph-PPR for external page-labeled benchmarks after the first transfer results underperformed `plain_top224` on page metrics.

## Short Conclusion

The first external-dataset transfer did not show that Graph-PPR is useless. It showed that the current best M3DocVQA config is solving the wrong objective for these benchmarks.

M3DocVQA/MMQA best config:

- optimized for document shortlist retrieval
- uses `--per-doc-page-limit 1`
- outputs one representative page per document
- strong for `doc@4`

External datasets:

- mostly evaluate exact page retrieval
- care whether the gold page itself is ranked early
- often already have high document recall

So the current `doc_shortlist_best` config can find the right document but throw away the gold page.

Main action:

- do not use `doc_shortlist_best` unchanged as the final page retriever
- use page-preserving Graph-PPR or Graph-PPR as a document prior combined with ColPali page scores

## Updated Page-Labeled Dataset Conclusion

Follow-up page-preserving runs show that the best general page-labeled config so far is:

```text
GRAPH_PROFILE=page_rank_probe
FINAL_TOP_PAGES=1000
PER_DOC_PAGE_LIMIT=0
DENSE_WEIGHT=1.25
SPARSE_WEIGHT=0.75
RESTART_PROB=0.15
PPR_ITERS=30
PAGE_DOC_EDGE_WEIGHT=1.0
SAME_DOC_WINDOW=1
ADJACENT_PAGE_EDGE_WEIGHT=0.25
FINAL_PAGE_SEED_WEIGHT=1.0
FINAL_PPR_PAGE_WEIGHT=0.25
FINAL_PPR_DOC_WEIGHT=0.25
```

Short label: `denseheavy_lightboth`.

This is better than the M3DocVQA `doc_shortlist_best` transfer and is generally better than `plain_top224` at practical page-retrieval depths, but not uniformly at rank 1. In particular, ViDoSeek remains a high-saturation case where `plain_top224` is slightly better at page@4.

| Dataset | page@1 | page@4 | page@20 | doc@4 | doc@20 | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| SciEGQA | 0.4951 *(plain 0.5228)* | 0.7686 *(plain 0.7394)* | 0.9104 *(plain 0.8758)* | 0.9261 *(plain 0.9070)* | 0.9852 *(plain 0.9772)* | wins at page@4/@20 and doc@4/@20; loses page@1 |
| MMDocIR | 0.4074 *(plain 0.4136)* | 0.6342 *(plain 0.6075)* | 0.7662 *(plain 0.7480)* | 0.8148 *(plain 0.8058)* | 0.8920 *(plain 0.8890)* | wins at page@4/@20 and doc@4/@20; loses page@1 |
| ViDoRe V3 | 0.1689 *(plain 0.1730)* | 0.3475 *(plain 0.3312)* | 0.5706 *(plain 0.5431)* | 0.8959 *(plain 0.8854)* | 0.9751 *(plain 0.9809)* | wins at page@4/@20 and doc@4; loses page@1/doc@20 |
| ViDoSeek | 0.6567 *(plain 0.6830)* | 0.8923 *(plain 0.8958)* | 0.9974 *(plain 0.9842)* | 1.0000 *(plain 0.9982)* | 1.0000 *(plain 1.0000)* | near-saturated; plain still slightly better at page@1/@4 |

Current claim:

- For page-labeled datasets, use page-preserving output: `PER_DOC_PAGE_LIMIT=0`, `FINAL_TOP_PAGES=1000`.
- Use `denseheavy_lightboth` as the strongest frozen general config unless the full sweep result collector identifies a clearly better single setting.
- Do not claim universal improvement over `plain_top224`: the gains are strongest at page@4/page@20, while page@1 often remains better for `plain_top224`.
- Keep `doc_shortlist_best` separate for M3DocVQA/MMQA-style document-shortlist retrieval.

## Transfer Results That Motivated This

Values outside parentheses are Graph-PPR `doc_shortlist_best`.
Values in parentheses are `plain_top224`.

| Dataset | qids | doc@1 | doc@4 | doc@20 | page@1 | page@4 | page@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MMDocIR | 1,658 | 0.6815 *(0.6852)* | 0.8034 *(0.8058)* | 0.8884 *(0.8890)* | 0.3861 *(0.4136)* | 0.4562 *(0.6075)* | 0.4998 *(0.7480)* |
| SciEGQA | 1,623 | 0.8355 *(0.8262)* | 0.9279 *(0.9070)* | 0.9846 *(0.9772)* | 0.4624 *(0.5228)* | 0.5173 *(0.7394)* | 0.5474 *(0.8758)* |
| ViDoSeek | 1,142 | 0.9860 *(0.9939)* | 0.9991 *(0.9982)* | 1.0000 *(1.0000)* | 0.6655 *(0.6830)* | 0.6743 *(0.8958)* | 0.6751 *(0.9842)* |
| ViDoRe V3 | 14,514 | 0.6521 *(0.6586)* | 0.8725 *(0.8854)* | 0.9703 *(0.9809)* | 0.1472 *(0.1730)* | 0.2064 *(0.3312)* | 0.2285 *(0.5431)* |

SciEGQA is the most diagnostic:

- Graph-PPR improves document retrieval:
  - `doc@1`: `0.8355` vs `0.8262`
  - `doc@4`: `0.9279` vs `0.9070`
  - `doc@20`: `0.9846` vs `0.9772`
- but it badly hurts page retrieval:
  - `page@20`: `0.5474` vs `0.8758`

This means the graph is helping identify documents, but the one-page-per-doc output is selecting the wrong page inside many correct documents.

## Why The Current Best Config Fails On Page Benchmarks

Current M3DocVQA best config:

```text
graph1000 / doc_shortlist_best
dense_top_pages = 1000
sparse_top_pages = 1000
final_top_pages = 20
per_doc_page_limit = 1
doc_seed_weight = 0.0
restart_prob = 0.15
ppr_iters = 30
final_page_seed_weight = 1.0
final_ppr_page_weight = 1.5
final_ppr_doc_weight = 0.75
```

Problem:

- `--per-doc-page-limit 1` keeps only one page from each document
- if Graph-PPR picks the wrong page from the correct document, doc metrics still look good
- page metrics count that as a miss

Additional issue:

- Graph-PPR smooths evidence across pages and documents
- this is good for document discovery
- it can blur the exact page-local evidence that ColPali `plain_top224` preserves

For ViDoRe / ViDoSeek:

- document recall is already high
- the real problem is exact page localization
- Graph-PPR adds little as a doc retriever and can hurt page ranking

For SciEGQA:

- Graph-PPR clearly improves document ranking
- therefore it should be used as a document prior or candidate generator
- it should not be the final page selection mechanism in one-page-per-doc mode

## Configs To Try Next

### 1. Page-Preserving Graph-PPR

This is the first fix to try.

Change:

```text
per_doc_page_limit = 0
final_top_pages = 1000
```

Keep the same Graph-PPR weights initially:

```bash
python scripts/graph_rerank_page_retrieval_predictions.py \
  --dense-prediction-json "${DENSE_PRED}" \
  --sparse-prediction-json "${SPLADE_PRED}" \
  --gold "${GOLD_JSONL}" \
  --question-type "${QUESTION_TYPE}" \
  --dense-top-pages 1000 \
  --sparse-top-pages 1000 \
  --final-top-pages 1000 \
  --per-doc-page-limit 0 \
  --rrf-k 10 \
  --dense-weight 1.0 \
  --sparse-weight 1.0 \
  --doc-seed-weight 0.0 \
  --restart-prob 0.15 \
  --ppr-iters 30 \
  --page-doc-edge-weight 1.0 \
  --same-doc-window 1 \
  --adjacent-page-edge-weight 0.25 \
  --final-page-seed-weight 1.0 \
  --final-ppr-page-weight 1.5 \
  --final-ppr-doc-weight 0.75 \
  --output-prediction-json "${OUTDIR}/graph_ppr_page_preserve_1000.prediction.json" \
  --output-summary-json "${OUTDIR}/graph_ppr_page_preserve_1000.summary.json"
```

Expected behavior:

- doc metrics may stay strong
- page@20 should recover because gold pages are no longer discarded
- page@1/page@4 may still need retuning because doc smoothing can still overpower page-local evidence

Run this before adding new code.

### 2. Page-Local Dominant Graph-PPR

For page-labeled datasets, the final score should keep ColPali/SPLADE page evidence dominant and use graph scores as a weak prior.

Try:

```text
final_page_seed_weight = 1.0
final_ppr_page_weight = 0.25 or 0.5
final_ppr_doc_weight = 0.10 or 0.25
per_doc_page_limit = 0
final_top_pages = 1000
```

Recommended small sweep:

| config | final_page_seed_weight | final_ppr_page_weight | final_ppr_doc_weight |
| --- | ---: | ---: | ---: |
| seed_plus_light_page | 1.0 | 0.25 | 0.0 |
| seed_plus_light_doc | 1.0 | 0.0 | 0.25 |
| seed_plus_light_both | 1.0 | 0.25 | 0.25 |
| seed_plus_medium_both | 1.0 | 0.5 | 0.25 |
| M3DocVQA_best_page_preserve | 1.0 | 1.5 | 0.75 |

Hypothesis:

- M3DocVQA best weights are too graph-heavy for exact page retrieval
- lighter PPR/doc weights should improve page metrics

### 3. Dense-Heavy Page-Preserving Graph-PPR

SPLADE is often useful for finding the right document family but weaker for exact page/layout selection.

Try dense-heavy source weights:

```text
dense_weight = 1.25
sparse_weight = 0.75
```

and optionally:

```text
dense_weight = 1.5
sparse_weight = 0.5
```

Keep:

```text
per_doc_page_limit = 0
final_top_pages = 1000
final_page_seed_weight = 1.0
final_ppr_page_weight = 0.25 or 0.5
final_ppr_doc_weight = 0.10 or 0.25
```

Hypothesis:

- dense ColPali page evidence should dominate page ranking
- SPLADE should act as recall support, not final page selector

### 4. Two-Stage Graph-PPR Doc Prior + ColPali Page Rerank

This is likely the strongest conceptual fix.

Stage 1:

- use Graph-PPR to rank documents
- keep top `K` docs, e.g. `20`, `50`, or `100`

Stage 2:

- within those docs, rank pages by original `plain_top224` / ColPali page score
- do not choose the Graph-PPR representative page

Expected behavior:

- preserves Graph-PPR document discovery gains
- preserves ColPali exact page-local ranking
- directly targets the SciEGQA failure mode

This may need a small helper script if not already implemented for the target dataset.

Recommended variants:

| doc candidates from Graph-PPR | final page scorer |
| --- | --- |
| top20 docs | plain_top224 page score |
| top50 docs | plain_top224 page score |
| top100 docs | plain_top224 page score |

Main metric to watch:

- page@4 and page@20 should recover
- doc@4 should stay near Graph-PPR

### 5. No Page-Doc Edge Control

Run this to verify whether graph propagation is the thing hurting page metrics:

```text
page_doc_edge_weight = 0.0
same_doc_window = 0
adjacent_page_edge_weight = 0.0
ppr_iters = 0
per_doc_page_limit = 0
final_top_pages = 1000
```

This is page-RRF in page-preserving mode.

If page-RRF is better than Graph-PPR on page@k, graph smoothing is too strong.

If page-RRF is also worse than `plain_top224`, then the dense+SPLADE rank fusion itself is hurting page ranking.

## Recommended Experiment Order

Run the smallest diagnostic set first.

1. `plain_top224`
   - existing baseline
2. current `doc_shortlist_best`
   - confirms the failure pattern
3. page-preserving Graph-PPR with M3DocVQA best weights
   - `per_doc_page_limit=0`, `final_top_pages=1000`
4. page-preserving page-RRF
   - no graph, no PPR, `per_doc_page_limit=0`
5. page-local dominant Graph-PPR
   - `page_seed=1.0`, `page_ppr=0.25`, `doc_ppr=0.25`
6. dense-heavy page-local dominant Graph-PPR
   - `dense_weight=1.25`, `sparse_weight=0.75`
7. two-stage Graph-PPR docs + ColPali page rerank
   - if steps 3-6 do not beat `plain_top224`

## How To Interpret Outcomes

### Case A: Page-preserving Graph-PPR beats `doc_shortlist_best` but not `plain_top224`

Conclusion:

- one-page-per-doc was a major problem
- graph smoothing is still too strong or not page-local enough
- try page-local dominant weights and dense-heavy source weights

### Case B: Page-preserving Graph-PPR beats `plain_top224`

Conclusion:

- Graph-PPR transfers after removing the doc-shortlist output constraint
- use this as the external-dataset config
- still report `doc_shortlist_best` only as a doc-retrieval config

### Case C: Page-preserving page-RRF beats Graph-PPR

Conclusion:

- dense+SPLADE page fusion helps
- graph propagation hurts exact page ranking
- use page-RRF or lighter PPR weights for page-labeled tasks

### Case D: Two-stage wins

Conclusion:

- Graph-PPR is best as a document prior
- ColPali remains best for exact page selection
- this is the strongest general recipe:
  - graph for doc discovery
  - ColPali for page localization

## Dataset-Specific Expectations

### SciEGQA

Most promising transfer target.

Reason:

- Graph-PPR already improved doc metrics
- page metrics failed because of wrong page selection

Best next bet:

- two-stage Graph-PPR docs + ColPali page rerank
- page-preserving Graph-PPR with light graph weights

### ViDoSeek

Document recall is almost saturated.

Reason:

- `doc@20 = 1.0000` for both methods
- page ranking is the real bottleneck

Best next bet:

- stay close to `plain_top224`
- only use very weak doc prior
- do not use one-page-per-doc Graph-PPR

### ViDoRe V3

Page localization is hard and important.

Reason:

- Graph-PPR page@20 is much worse than `plain_top224`
- doc@20 is also lower than `plain_top224`

Best next bet:

- page-preserving page-RRF control
- dense-heavy page-local dominant Graph-PPR
- two-stage doc prior only if Graph-PPR doc candidates improve candidate recall

### MMDocIR

Graph-PPR is slightly worse on both doc and page metrics.

Best next bet:

- do not assume graph helps
- first check page-preserving page-RRF
- then try dense-heavy light-prior Graph-PPR

## Reporting Recommendation

Use this wording:

```text
The M3DocVQA-tuned Graph-PPR config is a document-shortlist reranker. It improves document retrieval on M3DocVQA and SciEGQA, but its one-page-per-document output is not appropriate as a final page retriever for page-labeled benchmarks. For external page-retrieval datasets, the next configuration should preserve multiple pages per document and use graph scores as a weak document prior, or use Graph-PPR only for document selection followed by ColPali page-local reranking.
```

Do not claim:

- Graph-PPR failed generally
- SPLADE is useless
- graph methods cannot help page retrieval

Claim instead:

- `doc_shortlist_best` does not transfer as a final page retriever
- the transfer path should be page-preserving and page-local-score dominated

## Bottom Line For The Datasets Chat

The next chat should not rerun the same `doc_shortlist_best` config as the final benchmark method.

Priority configs:

1. page-preserving Graph-PPR:
   - `per_doc_page_limit=0`
   - `final_top_pages=1000`
2. light-prior Graph-PPR:
   - `page_seed=1.0`
   - `page_ppr=0.25`
   - `doc_ppr=0.25`
3. dense-heavy light-prior Graph-PPR:
   - `dense_weight=1.25`
   - `sparse_weight=0.75`
4. two-stage:
   - Graph-PPR top docs
   - ColPali / `plain_top224` page rerank inside those docs

The main target is to keep Graph-PPR's document discovery gains without discarding or demoting the exact gold page.

## SciEGQA Targeted Sweep Runner

A ready-to-run SciEGQA sweep is available:

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
git pull --rebase
bash scripts/run_sciegqa_page_preserving_graph_ppr_sweep.sh
```

It runs:

- `plain_top224` baseline evaluation
- 3 dense/SPLADE source weights:
  - `1.0 / 1.0`
  - `1.25 / 0.75`
  - `1.5 / 0.5`
- 6 page-preserving final-score settings:
  - page-RRF, no PPR
  - light page PPR only
  - light doc PPR only
  - light page+doc PPR
  - medium page+doc PPR
  - M3DocVQA-best weights in page-preserving mode
- doc-shortlist control with `GRAPH_PROFILE=doc_shortlist_best`

It saves recall tables:

```text
$LOCAL_OUTPUT_DIR/sciegqa/graph_ppr_plain_top224_splade/sciegqa_pagepreserve_sweep_recall_table.md
$LOCAL_OUTPUT_DIR/sciegqa/graph_ppr_plain_top224_splade/sciegqa_pagepreserve_sweep_recall_table.csv
```

Use page recall@4 as the primary selection metric, with page recall@1, @10, @20 and doc recall@4 as secondary checks.

The generic runner for other datasets is:

```bash
bash scripts/run_external_page_preserving_graph_ppr_sweep.sh
```

Set `DATA_NAME`, `DATA_ROOT`, `DENSE_PRED`, `SPARSE_PRED`, `OUT_DIR`, and `LABEL_PREFIX` before calling it.

### MMDocIR Sweep

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
git pull --rebase
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
source mmdocir/env_hpc.sh

DATA_NAME=mmdocir \
DATA_ROOT="$LOCAL_DATA_DIR/mm-docir" \
DENSE_PRED="$LOCAL_OUTPUT_DIR/mmdocir/plain_top224_ret1000_prediction.json" \
SPARSE_PRED="$LOCAL_OUTPUT_DIR/mmdocir/doc_rrf_exact_dense_splade/mmdocir_splade_ret1000.prediction.json" \
OUT_DIR="$LOCAL_OUTPUT_DIR/mmdocir/graph_ppr_plain_top224_splade" \
LABEL_PREFIX="mmdocir_pagepreserve_sweep" \
bash scripts/run_external_page_preserving_graph_ppr_sweep.sh
```

### ViDoRe V3 Sweep

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
git pull --rebase
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
unset HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME
source vidore/env_hpc.sh

DATA_NAME=vidore-v3 \
DATA_ROOT="$LOCAL_DATA_DIR/vidore-v3" \
DENSE_PRED="$LOCAL_OUTPUT_DIR/vidore-v3/plain_top224_ret1000_prediction.json" \
SPARSE_PRED="$LOCAL_OUTPUT_DIR/vidore-v3/doc_rrf_exact_dense_splade/vidore-v3_splade_ret1000.prediction.json" \
OUT_DIR="$LOCAL_OUTPUT_DIR/vidore-v3/graph_ppr_plain_top224_splade" \
LABEL_PREFIX="vidore-v3_pagepreserve_sweep" \
bash scripts/run_external_page_preserving_graph_ppr_sweep.sh
```

### ViDoSeek Sweep

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
git pull --rebase
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
source vidoseek/env_hpc.sh

DATA_NAME=vidoseek \
DATA_ROOT="$LOCAL_DATA_DIR/vidoseek" \
DENSE_PRED="$LOCAL_OUTPUT_DIR/vidoseek/plain_top224_ret1000_prediction.json" \
SPARSE_PRED="$LOCAL_OUTPUT_DIR/vidoseek/doc_rrf_plain_top224_splade/vidoseek_splade_ret1000.prediction.json" \
OUT_DIR="$LOCAL_OUTPUT_DIR/vidoseek/graph_ppr_plain_top224_splade" \
LABEL_PREFIX="vidoseek_pagepreserve_sweep" \
bash scripts/run_external_page_preserving_graph_ppr_sweep.sh
```

## Collect All Sweep Outputs Together

After the SciEGQA, MMDocIR, ViDoRe V3, and ViDoSeek sweeps finish, run:

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
git pull --rebase

"$REPO_ROOT/env/bin/python" scripts/collect_page_preserving_sweep_results.py \
  --dataset SciEGQA /mmfs1/scratch/jacks.local/aerfanshekooh/custom/SciEGQA_M3DocRAG/output/sciegqa/graph_ppr_plain_top224_splade/sciegqa_pagepreserve_sweep_recall_table.csv \
  --plain-eval SciEGQA /mmfs1/scratch/jacks.local/aerfanshekooh/custom/SciEGQA_M3DocRAG/output/sciegqa/graph_ppr_plain_top224_splade/sciegqa_pagepreserve_sweep_plain_top224.eval.txt \
  --dataset MMDocIR /mmfs1/scratch/jacks.local/aerfanshekooh/custom/MMDocIR_M3DocRAG/output/mmdocir/graph_ppr_plain_top224_splade/mmdocir_pagepreserve_sweep_recall_table.csv \
  --plain-eval MMDocIR /mmfs1/scratch/jacks.local/aerfanshekooh/custom/MMDocIR_M3DocRAG/output/mmdocir/graph_ppr_plain_top224_splade/mmdocir_pagepreserve_sweep_plain_top224.eval.txt \
  --dataset ViDoRe-V3 /mmfs1/scratch/jacks.local/aerfanshekooh/custom/ViDoRe_M3DocRAG/output/vidore-v3/graph_ppr_plain_top224_splade/vidore-v3_pagepreserve_sweep_recall_table.csv \
  --plain-eval ViDoRe-V3 /mmfs1/scratch/jacks.local/aerfanshekooh/custom/ViDoRe_M3DocRAG/output/vidore-v3/graph_ppr_plain_top224_splade/vidore-v3_pagepreserve_sweep_plain_top224.eval.txt \
  --dataset ViDoSeek /mmfs1/scratch/jacks.local/aerfanshekooh/custom/ViDoSeek_M3DocRAG/output/vidoseek/graph_ppr_plain_top224_splade/vidoseek_pagepreserve_sweep_recall_table.csv \
  --plain-eval ViDoSeek /mmfs1/scratch/jacks.local/aerfanshekooh/custom/ViDoSeek_M3DocRAG/output/vidoseek/graph_ppr_plain_top224_splade/vidoseek_pagepreserve_sweep_plain_top224.eval.txt \
  --top-n 5 \
  --sort-metric page@4 \
  | tee /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/pagepreserve_sweep_top5_by_page4.md
```

For only the winning config per dataset, change `--top-n 5` to `--top-n 1`.
