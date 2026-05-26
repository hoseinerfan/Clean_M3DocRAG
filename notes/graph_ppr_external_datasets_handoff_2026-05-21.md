# Graph-PPR External Dataset Handoff

Date: 2026-05-21

Purpose: use this note in the datasets chat to adjust Graph-PPR for external page-labeled benchmarks. The first direct transfer underperformed `plain_top224` on page metrics; follow-up page-preserving sweeps found a better page-labeled default.

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
- use `denseheavy125_medium_both` as the current best single page-labeled Graph-PPR default
- keep Graph-PPR as a document prior plus ColPali page scoring as the next major direction if more improvement is needed

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
FINAL_PPR_PAGE_WEIGHT=0.5
FINAL_PPR_DOC_WEIGHT=0.25
```

Short label: `denseheavy125_medium_both`.

This is better than the M3DocVQA `doc_shortlist_best` transfer and is better than `plain_top224` at practical page-retrieval depths on all four checked page-labeled datasets. ViDoSeek remains a high-saturation case; its best individual sweep row is `denseheavy150_m3best_pagepreserve`, but `denseheavy125_medium_both` is close and still beats `plain_top224` at page@4/page@20.

| Dataset | best sweep row by page@4 | page@1 | page@4 | page@20 | doc@4 | doc@20 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| SciEGQA | `denseheavy125_medium_both` | 0.5508 *(plain 0.5228)* | 0.8152 *(plain 0.7394)* | 0.9248 *(plain 0.8758)* | 0.9291 *(plain 0.9070)* | 0.9871 *(plain 0.9772)* | broad win except page@20/doc@20 still close to other graph variants |
| MMDocIR | `denseheavy125_medium_both` | 0.4596 *(plain 0.4136)* | 0.6719 *(plain 0.6075)* | 0.7889 *(plain 0.7480)* | 0.8160 *(plain 0.8058)* | 0.8938 *(plain 0.8890)* | clear page@1/@4/@20 win |
| ViDoRe V3 | `denseheavy125_medium_both` | 0.3902 *(plain 0.1730)* | 0.6465 *(plain 0.3312)* | 0.8227 *(plain 0.5431)* | 0.9099 *(plain 0.8854)* | 0.9788 *(plain 0.9809)* | large page gain; tiny doc@20 loss |
| ViDoSeek | `denseheavy150_m3best_pagepreserve` | 0.6909 *(plain 0.6830)* | 0.9037 *(plain 0.8958)* | 0.9982 *(plain 0.9842)* | 0.9991 *(plain 0.9982)* | 1.0000 *(plain 1.0000)* | saturated; best row uses heavier graph weights |

Current claim:

- For page-labeled datasets, use page-preserving output: `PER_DOC_PAGE_LIMIT=0`, `FINAL_TOP_PAGES=1000`.
- Use `denseheavy125_medium_both` as the strongest frozen single general config for page-labeled datasets.
- Treat `denseheavy150_m3best_pagepreserve` as a ViDoSeek-specific best row, not the global default.
- Do not collapse the result to one metric: the strongest and most stable gains are at page@4/page@20, and page@1 should still be reported separately.
- Keep `doc_shortlist_best` separate for M3DocVQA/MMQA-style document-shortlist retrieval.
- MMLongBench DocQA is now prepared as the next page-labeled stress test. It should use the same page-preserving default first, because `ans_page_list` provides exact zero-based page labels.
- DUDE is scaffolded as the MP-DocVQA replacement target. It should also use the page-preserving default first, after validating the converter's `answer_page_base` sanity summary.

## 2026-05-23 Addendum: Graph Augmentation Status

Recent experiments added three new graph-augmentation directions: PDF hyperlink edges for M3DocVQA/MMQA, query-anchor/financial evidence audits for MMDocIR, and LayoutLMv3 page-embedding kNN edges for MMDocIR.

### M3DocVQA PDF hyperlink graph

The PDF annotations expose a real Wikipedia hyperlink graph:

```text
valid docs checked: 3366
deduped edges: 21451
source doc coverage: 2885 / 3366 = 85.71%
target doc coverage: 2417 / 3366 = 71.81%
```

This is useful because many M3DocVQA questions are entity-chain or bridge-document questions. Hyperlinks provide authored edges from a retrieved source page to a related target document, especially for ImageListQ and visual/entity questions.

Best current fixed-weight results against `fulldev_nohyperlink_denseheavy125_medium_both`:

| Run | doc@1 | doc@4 | doc@20 | doc@100 | Movement summary |
| --- | ---: | ---: | ---: | ---: | --- |
| no hyperlink baseline | 0.6080 | 0.8458 | 0.9272 | 0.9632 | baseline |
| hyperlink `w0p05` | 0.6099 | 0.8479 | 0.9283 | 0.9655 | 48 improved / 22 worsened |
| hyperlink `w0p10` | 0.6090 | 0.8517 | 0.9292 | 0.9684 | 63 improved / 36 worsened |
| hyperlink `w0p20` | 0.6057 | 0.8527 | 0.9305 | 0.9726 | 78 improved / 53 worsened |

Interpretation:

- fixed `w0p10` is the best balanced headline setting so far
- fixed `w0p20` gives the highest doc@4/doc@100 but hurts doc@1
- source/target gated query-supported `w0p10` is safer at doc@1 but lower at doc@4
- source-target rank decay over-penalizes useful Wikipedia jumps and should not be used as a main setting

Hyperlink audit:

```text
gold_has_any_incoming_link_count: 1959 / 2441
gold_linked_from_baseline_sources_count: 1932 / 2441
```

The largest gains are concentrated in `ImageListQ`, where retrieved bridge pages often link to the target entity/document needed for a visual answer. This should be framed as a real authored document graph, not as a hand-built heuristic.

### MMDocIR query-anchor and financial evidence

Query-anchor evidence remains the strongest graph-native MMDocIR augmentation, but the audits show why domain-specific gains saturate.

Financial subset:

```text
qids: 344
best broad financial verifier recovered/lost: 3 / 0
main limitations:
  gold_page_no_financial_evidence_match: 187
  evidence_too_broad: 87
  gold_page_has_evidence_but_not_in_candidate_head: 29
  gold_page_missing_from_candidate_pool: 18
```

The financial verifier can catch true table-like evidence, but naive financial text matching is too broad: many pages contain the same metric/year tokens. A strict doc-prior version reduced broad positive pages but caused 2 lost top-4 cases, so it should be treated as a diagnostic/ablation, not a headline method yet.

News subset:

```text
qids: 137
recovered: 2
improved_rank: 38
worsened_rank: 29
missing_in_both: 14
main limitation:
  competing_top_pages_match_as_many_or_more_anchors: 45
```

The News audit is useful because it isolates the weakness of pure anchor matching: topical/entity redundancy. The right next step is not stronger anchor weight; it is learned or verifier-based evidence calibration, or at least multi-anchor relation/co-occurrence verification.

### LayoutLMv3 page-kNN graph

The LayoutLMv3 experiment should currently be reported as a negative ablation:

```text
embedded pages: 20214
source_counts: {'fallback_text': 20214}
```

Because all pages used fallback text rather than real OCR boxes/layout inputs, the graph was essentially a text-embedding kNN graph, not a true layout graph.

Observed behavior:

- broad cross-doc kNN hurt MMDocIR page@4: `1114 -> 1104`
- same-doc kNN was neutral/slightly worse: page@4 stayed around `1114`, doc@4 dropped slightly
- gated same-doc top1 also did not help: recovered/lost `0 / 2`

Conclusion: do not use fallback LayoutLMv3 kNN as a main novelty result. Revisit only with real OCR boxes/layout patches or a DocGraphLM-style page representation.

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
- the old one-page-per-doc Graph-PPR adds little as a doc retriever and can hurt page ranking

For SciEGQA:

- Graph-PPR clearly improves document ranking
- therefore it should be used as a document prior or candidate generator
- it should not be the final page selection mechanism in one-page-per-doc mode

## Confirmed Config And Remaining Controls

### 1. Confirmed default: `denseheavy125_medium_both`

This is the best general page-labeled Graph-PPR config found so far.

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
FINAL_PPR_PAGE_WEIGHT=0.5
FINAL_PPR_DOC_WEIGHT=0.25
```

Equivalent direct graph command:

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
  --dense-weight 1.25 \
  --sparse-weight 0.75 \
  --doc-seed-weight 0.0 \
  --restart-prob 0.15 \
  --ppr-iters 30 \
  --page-doc-edge-weight 1.0 \
  --same-doc-window 1 \
  --adjacent-page-edge-weight 0.25 \
  --final-page-seed-weight 1.0 \
  --final-ppr-page-weight 0.5 \
  --final-ppr-doc-weight 0.25 \
  --output-prediction-json "${OUTDIR}/graph_ppr_denseheavy125_medium_both.prediction.json" \
  --output-summary-json "${OUTDIR}/graph_ppr_denseheavy125_medium_both.summary.json"
```

Why this works better than the M3DocVQA config:

- it preserves pages instead of keeping only one page per doc
- it keeps dense ColPali page evidence stronger than SPLADE
- it keeps graph/PPR controlled: a moderate page prior plus a light document prior, without letting graph smoothing dominate exact page evidence

Use this as the default page-labeled benchmark config.

### 2. Remaining useful controls

Still run these when validating a new dataset:

- `plain_top224`
  - strongest old baseline
- `doc_shortlist_best`
  - shows why the M3DocVQA doc-shortlist config is not a final page retriever
- page-preserving page-RRF
  - no graph / no PPR control
- M3DocVQA-best weights in page-preserving mode
  - checks whether the issue was only `PER_DOC_PAGE_LIMIT=1` or also graph-heavy scoring
- two-stage Graph-PPR docs + ColPali page rerank
  - next direction if `denseheavy125_medium_both` is not enough

### 3. Page-local dominant sweep history

The sweep that led to `denseheavy125_medium_both` tested:

| config | final_page_seed_weight | final_ppr_page_weight | final_ppr_doc_weight |
| --- | ---: | ---: | ---: |
| seed_plus_light_page | 1.0 | 0.25 | 0.0 |
| seed_plus_light_doc | 1.0 | 0.0 | 0.25 |
| seed_plus_light_both / `lightboth` | 1.0 | 0.25 | 0.25 |
| seed_plus_medium_both / `medium_both` | 1.0 | 0.5 | 0.25 |
| M3DocVQA_best_page_preserve | 1.0 | 1.5 | 0.75 |

Best source weights for the general page-labeled setting:

- `dense_weight = 1.25`
- `sparse_weight = 0.75`

Interpretation:

- M3DocVQA best weights were too graph-heavy for exact page retrieval
- equal dense/sparse source weighting underused ColPali page-local evidence
- dense-heavy + medium page / light doc PPR is the best current single compromise

### 4. Two-Stage Graph-PPR Doc Prior + ColPali Page Rerank

This remains the strongest conceptual follow-up if `denseheavy125_medium_both` plateaus.

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

For a new page-labeled dataset, run this order:

1. `plain_top224`
   - baseline
2. `denseheavy125_medium_both`
   - current best general Graph-PPR page-labeled config
3. `doc_shortlist_best`
   - diagnostic only; confirms why the M3DocVQA doc-shortlist config should not be used as a page retriever
4. page-preserving page-RRF
   - no graph / no PPR control
5. M3DocVQA-best weights in page-preserving mode
   - checks whether the old failure was only the one-page-per-doc output or also graph-heavy scoring
6. two-stage Graph-PPR docs + ColPali page rerank
   - next major direction if `denseheavy125_medium_both` is not enough

## How To Interpret Outcomes

### Case A: `denseheavy125_medium_both` beats `plain_top224` at page@4/page@20

Conclusion:

- page-preserving dense-heavy medium-page/light-doc Graph-PPR transfers
- use it as the external page-labeled default
- still report page@1 separately because rank-1 behavior can differ from page@4/page@20

### Case B: `denseheavy125_medium_both` improves docs but not pages

Conclusion:

- Graph-PPR is useful as a document prior
- use the two-stage Graph-PPR docs + ColPali page rerank direction

### Case C: Page-preserving page-RRF beats Graph-PPR

Conclusion:

- dense+SPLADE page fusion helps
- graph propagation hurts exact page ranking
- use page-RRF or make graph weights even lighter for that dataset

### Case D: Two-stage wins

Conclusion:

- Graph-PPR is best as a document prior
- ColPali remains best for exact page selection
- this is the strongest general recipe:
  - graph for doc discovery
  - ColPali for page localization

## Dataset-Specific Expectations

### SciEGQA

`denseheavy125_medium_both` is a strong setting here:

- page@4 improves over `plain_top224`
- page@20 improves over `plain_top224`
- doc@4 and doc@20 improve over `plain_top224`
- page@1 also improves in the full sweep result

Interpretation:

- page-preserving output fixed the major failure
- dense-heavy medium-page/light-doc graph prior is better than the old doc-shortlist config

### ViDoSeek

High-saturation case.

`denseheavy125_medium_both`:

- improves page@20 over `plain_top224`
- reaches perfect doc@4/doc@20
- the best individual row improves page@1/page@4 over `plain_top224`

Interpretation:

- use `denseheavy125_medium_both` as the single general config
- use `denseheavy150_m3best_pagepreserve` if optimizing ViDoSeek alone

### ViDoRe V3

`denseheavy125_medium_both`:

- improves page@4 and page@20 over `plain_top224`
- improves doc@4 over `plain_top224`
- loses slightly at page@1 and doc@20

Interpretation:

- the page-preserving dense-heavy medium-page/light-doc fix works
- rank-1 exact page precision still favors `plain_top224`

### MMDocIR

`denseheavy125_medium_both`:

- improves page@4/page@20 over `plain_top224`
- improves doc@4/doc@20 over `plain_top224`
- loses slightly at page@1

Interpretation:

- do not use the old `doc_shortlist_best`
- use `denseheavy125_medium_both` as the Graph-PPR page-labeled default

## Reporting Recommendation

Use this wording:

```text
The M3DocVQA-tuned Graph-PPR config is a document-shortlist reranker and should not be used unchanged for page-labeled benchmarks. For external page-retrieval datasets, the best current single Graph-PPR setting is the page-preserving dense-heavy medium-page/light-doc prior config (`denseheavy125_medium_both`): it keeps all pages, weights ColPali dense evidence above SPLADE, and uses PPR as a moderate page prior plus light document prior. This recovers the page metrics that the one-page-per-doc config destroyed, especially at page@4 and page@20, while page@1 should still be reported separately.
```

Do not claim:

- Graph-PPR failed generally
- SPLADE is useless
- graph methods cannot help page retrieval

Claim instead:

- `doc_shortlist_best` does not transfer as a final page retriever
- `denseheavy125_medium_both` is the current best frozen single Graph-PPR config for page-labeled datasets
- the remaining transfer path should keep page output page-preserving and page-local-score dominated

## Bottom Line For The Datasets Chat

The next chat should not rerun the same `doc_shortlist_best` config as the final benchmark method.

Priority configs:

1. `denseheavy125_medium_both`:
   - `GRAPH_PROFILE=page_rank_probe`
   - `FINAL_TOP_PAGES=1000`
   - `PER_DOC_PAGE_LIMIT=0`
   - `dense_weight=1.25`
   - `sparse_weight=0.75`
   - `final_page_seed_weight=1.0`
   - `final_ppr_page_weight=0.5`
   - `final_ppr_doc_weight=0.25`
2. `plain_top224`:
   - required baseline
   - still competitive or better at page@1
3. page-preserving page-RRF:
   - no graph / no PPR control
4. M3DocVQA-best page-preserving control:
   - confirms why controlled graph weights are needed
5. two-stage:
   - Graph-PPR top docs
   - ColPali / `plain_top224` page rerank inside those docs

The main target is now to keep `denseheavy125_medium_both` as the frozen general page-labeled config, then test whether the two-stage design can improve rank-1 page precision.

## MMLongBench DocQA Next Run

MMLongBench support is scaffolded under `mmlongbench/`. The first target is the DocQA subset (`longdocurl`, `mmlongdoc`, `slidevqa`) because it exposes `ans_page_list` and can be evaluated with the same exact page metrics as MMDocIR, SciEGQA, ViDoRe V3, and ViDoSeek.

Recommended first run:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
source mmlongbench/env_hpc.sh

"$REPO_ROOT/env/bin/python" mmlongbench/prepare_mmlongbench.py \
  --download \
  --snapshot-dir "$MMLONGBENCH_WORK_ROOT/hf_snapshot/MMLongBench" \
  --output-root "$LOCAL_DATA_DIR/mmlongbench-docqa"

sbatch --time=12:00:00 --array=0-31 --export=ALL,NUM_SHARDS=32,BATCH_SIZE=2 \
  mmlongbench/sbatch_embed_mmlongbench_array.sh
```

After dense retrieval and `plain_top224`, use `denseheavy125_medium_both` as the first Graph-PPR config. Full commands are in `mmlongbench/README.md` and `DATASET_WORKFLOWS.md`.

## DUDE Status

DUDE support is implemented under `dude/`. Use it as the MP-DocVQA replacement path because it is multi-page DocQA, exposes PDFs/OCR, and has answer page bounding boxes that can be converted into exact page retrieval labels.

Current status: DUDE is prepared, embedded/indexed, and dense baseline retrieval is complete. The dense baseline artifact is:

```text
/mmfs1/scratch/jacks.local/aerfanshekooh/custom/DUDE_M3DocRAG/output/dude/baseline_ret1000.json
```

Observed dense baseline:

```text
n_qids=2903
page_recall@4=0.5354403490641865
page_recall@20=0.6543920082673097
doc_recall@4=0.6114364450568378
doc_recall@20=0.7247674819152601
page_hit@4=1565
doc_hit@4=1775
```

The `plain_top224` artifact is:

```text
/mmfs1/scratch/jacks.local/aerfanshekooh/custom/DUDE_M3DocRAG/output/dude/plain_top224_ret1000_prediction.json
```

Observed `plain_top224`:

```text
n_qids=2903
page_recall@4=0.5719543001492708
page_recall@20=0.6866134649978949
doc_recall@4=0.6507061660351361
doc_recall@20=0.7561143644505683
page_hit@4=1672
doc_hit@4=1889
improved_doc_rank_count=781
```

The SPLADE/doc-RRF artifacts are:

```text
/mmfs1/scratch/jacks.local/aerfanshekooh/custom/DUDE_M3DocRAG/output/dude/doc_rrf_plain_top224_splade/dude_splade_ret1000.prediction.json
/mmfs1/scratch/jacks.local/aerfanshekooh/custom/DUDE_M3DocRAG/output/dude/doc_rrf_plain_top224_splade/dude_exact_dense_splade_doc_rrf.prediction.json
```

Observed SPLADE/doc-RRF:

```text
n_qids=2903
page_recall@4=0.5311861292915375
page_recall@20=0.6180139319477934
doc_recall@4=0.6631071305545987
doc_recall@20=0.7747158112297623
page_hit@4=1558
doc_hit@4=1925
```

The Graph-PPR artifact is:

```text
/mmfs1/scratch/jacks.local/aerfanshekooh/custom/DUDE_M3DocRAG/output/dude/graph_ppr_plain_top224_splade/dude_denseheavy125_medium_both.prediction.json
```

Observed Graph-PPR:

```text
n_qids=2903
page_recall@4=0.5881731542082902
page_recall@20=0.7172139931871244
doc_recall@4=0.6796417499138822
doc_recall@20=0.7833275921460559
page_hit@4=1714
doc_hit@4=1973
```

Environment reset:

```bash
unset LOCAL_DATA_DIR LOCAL_EMBEDDINGS_DIR LOCAL_OUTPUT_DIR
unset HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE XDG_CACHE_HOME
source dude/env_hpc.sh
```

DUDE is complete through the frozen `denseheavy125_medium_both` page-labeled Graph-PPR config.

## Safe Heading/Bodyguard Gate Status

Current method name: **Boundary-Aware Multi-View Heading Rescue Gate**.

Use this as a conservative post-processing/rescue layer, not as the final global reranker. It only swaps a promoted page into the top 4 when:

- the candidate promotes a page from a narrow rescue window;
- the promoted document is already in the base top-4 documents;
- multiple heading views support the promoted page;
- the promoted page beats the displaced rank-boundary page by heading score;
- the promoted page passes a body-evidence guard;
- the query is not a layout-sensitive row/column/right/left query.

Validated safe-gate results so far:

| Dataset | accepted | base page hit@4 | candidate page hit@4 | gated page hit@4 | recovered | lost | net | body rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MMDocIR | 38 | 1114 | 1113 | 1117 | 3 | 0 | +3 | 25 |
| SciEGQA-Bench | 28 | 1323 | 1328 | 1323 | 0 | 0 | 0 | 23 |
| ViDoSeek | 45 | 1023 | 1033 | 1029 | 6 | 0 | +6 | 30 |
| DUDE (`doc-rank-1` gate) | 6 | 1733 | 1730 | 1733 | 0 | 0 | 0 | 1 |
| ViDoRe V3 (`text-heading` no-op) | 0 | 9383 | 9383 | 9383 | 0 | 0 | 0 | 0 |

Artifact paths:

```text
/mmfs1/scratch/jacks.local/aerfanshekooh/custom/MMDocIR_M3DocRAG/output/mmdocir/heading_breadcrumb_pdf_markdown_source_ablation/mmdocir_heuristic_strict_safe_gate_bodyguard.summary.json
/mmfs1/scratch/jacks.local/aerfanshekooh/custom/SciEGQA_M3DocRAG/output/sciegqa/heading_breadcrumb_pdf_markdown_source_ablation/sciegqa_safe_gate_bodyguard.summary.json
/mmfs1/scratch/jacks.local/aerfanshekooh/custom/ViDoSeek_M3DocRAG/output/vidoseek/heading_breadcrumb_pdf_markdown_source_ablation/vidoseek_strict_support_gate_layoutblock_no_page0_bodyguard.summary.json
/mmfs1/scratch/jacks.local/aerfanshekooh/custom/DUDE_M3DocRAG/output/dude/heading_breadcrumb_pdf_markdown_source_ablation/dude_safe_gate_bodyguard_docrank1.summary.json
/mmfs1/scratch/jacks.local/aerfanshekooh/custom/ViDoRe_M3DocRAG/output/vidore-v3/heading_breadcrumb_text_source_ablation/vidore_safe_gate_bodyguard.summary.json
```

DUDE note: the broad doc-top4 gate had one cross-document loss on a generic annual-report/year query. The `doc-rank-1` variant removes that loss and makes DUDE a neutral abstention result.

ViDoRe V3 note: the text-source Markdown variant preparation produced `0` outline heading lines,
`0` heuristic heading lines, and `0` strict heuristic heading lines. Consequently, the full,
heuristic-only, and strict heading graph outputs are identical to the no-heading control; the gate
accepts `0` promotions because there is no heading signal to verify.

Runner for reproducing or extending the M3DocVQA diagnostic:

```bash
DATASETS="m3docvqa" \
bash examples/run_safe_heading_gate_selected_datasets.sh
```

M3DocVQA has now been run as a document-only and page-0 proxy diagnostic. The runner expects
`plain_top224` and SPLADE predictions to exist first; if any prerequisite is missing, it prints
the missing path and stops.

Full rank-window rescue profile:

```bash
SAFE_GATE_PROFILE=window20 \
RUN_GOLD_RANK_AUDIT=1 \
DATASETS="m3docvqa" \
bash examples/run_safe_heading_gate_selected_datasets.sh
```

This keeps the same heading/body/layout/doc-rank safety checks but scans candidate ranks through
20 and only promotes pages whose base rank is in `5-20`. The default profile remains `boundary`
for reproducing the validated rank-5 artifacts. With `RUN_GOLD_RANK_AUDIT=1`, each dataset also
gets a `*.gold_rank_positions.md` report containing first gold page/doc ranks and a page-rank-band
by doc-rank-band matrix.

Current M3DocVQA path sanity from `scripts/discover_hpc_vital_paths.py`:

```text
M3DOCVQA_DENSE_PRED=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json
M3DOCVQA_SPARSE_PRED=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_splade_mmqa_dev/mmqa_dev_splade.prediction.json
M3DOCVQA_PAGE_TEXT_JSONL=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_page_text/m3docvqa_dev_page_text.jsonl
```

Use the raw `mmqa_dev_splade.prediction.json` for the safe-gate run. Historical
`fulldev_graph_ppr_sourceablate_no_splade` outputs are graph diagnostics and are not valid SPLADE
inputs here.

M3DocVQA has supporting document labels but no true supporting page indices in `MMQA_dev.jsonl`.
Consequently the safe-gate page-hit and recovered/lost metrics are not measurable on this dataset.
The runner now reports those fields as unavailable, prints document recall for retrieval sanity,
and skips the page-position gold audit. Evaluate the promoted page set through downstream VQA
before treating this dataset as positive or negative evidence for page rescue.

An `ImageListQ` page-0 proxy diagnostic is available for analysis only. It assumes page index `0`
of every supporting document is a relevant page, compares the safe gate with its no-heading
control, and reports all values under `synthetic_page_*` names so they are not confused with true
gold-page metrics:

```bash
export M3DOCVQA_HEADING_OUT="$PWD/output/m3docvqa_heading_breadcrumb_pdf_markdown_source_ablation"
"$PWD/env/bin/python" scripts/evaluate_first_page_gold_retrieval.py \
  --baseline-pred "$M3DOCVQA_HEADING_OUT/m3docvqa_heading_control_no_heading.prediction.json" \
  --pred "$M3DOCVQA_HEADING_OUT/m3docvqa_safe_window20_gate_bodyguard.prediction.json" \
  --gold "$PWD/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl" \
  --question-type ImageListQ \
  --first-page-idx 0 \
  --hit-k 4 \
  --recall-k 1 2 4 5 10 20 \
  --output-json "$M3DOCVQA_HEADING_OUT/m3docvqa_safe_window20_gate_bodyguard.imagelistq_page0_proxy.json"
```

Completed M3DocVQA proxy results (`ImageListQ`, `n=141`):

| Markdown source | heading pages | accepted | control synthetic hit@4 | gated synthetic hit@4 | recovered | lost | net | control doc hit@4 | gated doc hit@4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native PDF headings | 30,343 | 1,152 | 39 | 40 | 2 | 1 | +1 | 81 | 81 |
| `pymupdf4llm==0.3.4` | 25,355 | 766 | 43 | 42 | 1 | 2 | -1 | 87 | 87 |

Within the PyMuPDF4LLM full-dev run, direct strict-heading graph output raises document hit@4
from `2,346` to `2,349`, but the safe gate intentionally preserves the no-heading document
selection at `2,346`. There is no annotated page metric with which to judge its `766` accepted
page promotions. Under the page-0 proxy, the native extraction is mildly positive but not
loss-free, while PyMuPDF4LLM is negative at synthetic hit@4.

Do not compare the native and PyMuPDF4LLM control columns as a clean extraction ablation yet. The
no-heading controls already differ before page rescue, including full-dev document hit@4
(`2,279` native versus `2,346` PyMuPDF4LLM). Since a no-heading output should be independent of
heading extraction, rerun the native control and gate under the current graph inputs/code revision
before attributing this difference to Markdown conversion.

The next backend test should use an exact-page dataset rather than another M3DocVQA proxy. The
selected-dataset runner now supports `PDF_MARKDOWN_BACKEND=pymupdf4llm DATASETS="vidoseek"` and
`DATASETS="sciegqa"`. Start with ViDoSeek because its native safe gate is positive and loss-free
(`+6` at page hit@4) and its `5,349` pages keep extraction inexpensive. The ViDoSeek runner
preserves the frozen page-0 abstention rule and writes:

```text
/mmfs1/scratch/jacks.local/aerfanshekooh/custom/ViDoSeek_M3DocRAG/output/vidoseek/heading_breadcrumb_pdf_markdown_pymupdf4llm_source_ablation/vidoseek_safe_gate_bodyguard_no_page0.summary.json
```

```bash
PDF_MARKDOWN_FORCE_REBUILD=1 \
PDF_MARKDOWN_BACKEND=pymupdf4llm \
SAFE_GATE_PROFILE=boundary \
RUN_GOLD_RANK_AUDIT=1 \
DATASETS="vidoseek" \
bash examples/run_safe_heading_gate_selected_datasets.sh
```

Alternative M3DocVQA Markdown extraction experiment:

The exporter and selected-dataset runner support `PDF_MARKDOWN_BACKEND=pymupdf4llm`. The
HPC-safe comparison uses legacy PyMuPDF4LLM `0.3.4`, disables OCR controls exposed by its API,
and is a native-PDF structured-Markdown comparison rather than an OCR experiment. It writes a
separate output tree with suffix `_pymupdf4llm_source_ablation`. Current March 2026 releases
auto-activate ONNX-based Layout during import and emit invalid CPU-affinity errors under the
current SLURM binding; evaluate that neural-layout backend only as a separate configured run.
Installing the pinned converter may still update PyMuPDF in the shared environment, so keep the
frozen native JSONL artifacts and record package versions before claiming exact reproduction.

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
source hpc_vital_paths.generated.env
source scripts/m3docvqa_internal_env.sh
"$PWD/env/bin/python" -m pip install --force-reinstall "pymupdf4llm==0.3.4"

export ALT_DIR="$LOCAL_OUTPUT_DIR/m3docvqa_heading_breadcrumb_pdf_markdown_pymupdf4llm_source_ablation"
mkdir -p "$ALT_DIR"
"$PWD/env/bin/python" scripts/export_pdf_page_markdown.py \
  --doc-pages-jsonl "$M3DOCVQA_PAGE_TEXT_JSONL" \
  --pdf-root "$DATASET_ROOT" \
  --backend pymupdf4llm \
  --output-jsonl "$ALT_DIR/doc_pages_dev_with_pdf_markdown.jsonl" \
  --output-summary-json "$ALT_DIR/pdf_markdown_summary.json" \
  --progress-every 1000 \
  --body-char-limit 6000 \
  --require-heading-pages
"$PWD/env/bin/python" scripts/prepare_pdf_markdown_variants.py \
  --input-jsonl "$ALT_DIR/doc_pages_dev_with_pdf_markdown.jsonl" \
  --output-dir "$ALT_DIR/pdf_markdown_variants"
```

The completed PyMuPDF4LLM extraction is under
`/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output/m3docvqa_heading_breadcrumb_pdf_markdown_pymupdf4llm_source_ablation`.
It contains `44,294` pages, of which `25,355` contain headings, with `49,704` raw and `49,136`
strict heuristic heading lines. More heading lines are not sufficient evidence of better
Markdown: the paired page-0 proxy above is negative for the PyMuPDF4LLM safe gate.

```bash
PDF_MARKDOWN_FORCE_REBUILD=1 \
PDF_MARKDOWN_BACKEND=pymupdf4llm \
SAFE_GATE_PROFILE=window20 \
RUN_GOLD_RANK_AUDIT=1 \
DATASETS="m3docvqa" \
bash examples/run_safe_heading_gate_selected_datasets.sh
```

`PDF_MARKDOWN_FORCE_REBUILD=1` is required after any interrupted pre-fix extraction, because an
older run may have left a partial JSONL at the completed artifact path. Updated exports write
through `.tmp` files and move them into place only after conversion completes.

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
