# Visual Reranker Handoff (2026-04-27)

## Scope

This note summarizes the current state of the standalone visual-aware reranker in:

- `scripts/rerank_target_docs_visual_aware.py`

The baseline retrieval / FAISS / RAG code was not modified. All changes were kept inside the helper.

## Important code changes already pushed

These commits are already on `main`:

- `c18e5a8` Add target-page visual-aware rerank helper
- `2512ce9` Fix helper `torch` import in score-mask path
- `921f4e1` Fix helper `torch` import in channel-score path
- `2c25dd0` Add manual `--gold-page-uid` support
- `1bf0495` Make helper independent of plotting-script imports

On HPC, pull with:

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
git pull --rebase --autostash origin main
```

## Helper behavior

The helper reranks a fixed page pool offline. It does not change baseline retrieval.

Main inputs:

- `--baseline-pred ... --from-baseline-top-pages 1000`
- `--splice-query-token-labels ...`
- `--splice-patch-labels-jsonl ...`

Useful modes:

- doc-level supervision:
  - `--gold-doc-id ...`
  - or implicit gold docs from `supporting_context`
- page-level supervision:
  - `--gold-page-uid <doc_id>_page<idx>`

Main page features:

- `base_page_score`
- `visual_page_score`
- `non_visual_page_score`
- `balance_score`

Fused score:

```text
base * w_base + visual * w_visual + non_visual * w_non_visual + balance * w_balance
```

## Patch-label file situation

Old file used in many earlier runs:

- `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/layout_patch_assignments_done_so_far.jsonl`

Best current file:

- `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/layout_patch_assignments_done_so_far_plus_new_3class_full.jsonl`

The new file should be used for current reranker experiments. It fixed missing patch coverage for at least one key gold doc.

## Key qid: 46a4103ba65b176fba9ed85889775f8d

Question:

- `Which candidate has a mustache among the candidates in Delaware's Mini-Tuesday?`

Gold answer:

- `Al Sharpton`

Gold supporting doc:

- `5d6f8563f83fcc65dda2090b102cbc8c`

Manual audit:

- `page17` is an answer-bearing portrait page of Sharpton
- `page0` is also an answer-bearing page
- so this is a multi-valid-page case inside the gold doc

### Baseline ret1000 drop_pad_like

Prediction file:

- `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output/retrieval_only_dev_ret1000full_drop_pad_like/colpali-v1.2_ivfflat_ret1000_qtf-drop_pad_like_2026-04-16_07-21-44.json`

Gold-page hits inside top-1000 page rows:

- `page17` at row rank `13`, score `3.9078786373138428`
- `page29` at row rank `14`
- `page20` at row rank `22`
- `page30` at row rank `30`
- `page28` at row rank `51`
- `page16` at row rank `198`
- `page0` at row rank `858`, score `1.0452184677124023`

### Gold-doc-only diagnostics with new patch-label file

Output:

- `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/46a4103b_gold_doc_only_scores_plus_new_3class_full.json`

Important findings:

- new patch labels are active
- many pages now have nonzero `visual_patch_count` and `non_visual_patch_count`
- best page by base score:
  - `page28`
- best page by visual score:
  - `page17`

Examples:

- `page17`
  - `base=9.9112`
  - `visual=2.1630`
  - `non_visual=6.4490`
  - `balance=0.4299`
- `page0`
  - `base=8.4427`
  - `visual=1.9368`
  - `non_visual=5.3293`
  - `balance=0.3553`

### Page17-targeted rerank

Command used conceptually:

- page pool = baseline top-1000 pages
- target page = `5d6f8563f83fcc65dda2090b102cbc8c_page17`
- patch labels = `...plus_new_3class_full.jsonl`

Output:

- `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/46a4103b_rerank_page17.json`

Best result so far for `page17`:

- `first_gold_page_rank = 6`
- `first_gold_doc_rank = 4`

Chosen weights:

- `base = 1.0`
- `visual = 2.0`
- `non_visual = 1.0`
- `balance = 2.0`

Interpretation:

- baseline global page-row rank for `page17` was `13`
- reranking improved it to page rank `6`
- doc-level gold rank reached top-4

### Page0-targeted rerank

Output:

- `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/46a4103b_rerank_page0.json`

Best result so far for `page0`:

- `first_gold_page_rank = 83`
- `first_gold_doc_rank = 7`

Chosen weights:

- `base = 1.0`
- `visual = 2.0`
- `non_visual = 0.0`
- `balance = 2.0`

Interpretation:

- baseline global page-row rank for `page0` was `858`
- reranking improved it to `83`
- still far from top-4 pages

## Important interpretation for 46a4103b...

This qid is no longer a clean single-page failure case.

What it is good for:

- showing that page-level reranking can improve a real answer-bearing page
- showing that patch coverage matters
- showing that page17 is much easier for the current feature family than page0

What it is not good for:

- proving exact single-page recovery

## Other qids and findings

### e783cba0b3df36372d11823e378e5437

LGBT question:

- gold supporting doc id:
  - `d57e56eff064047af5a6ef074a570956`

Important earlier result:

- injecting `page0` into the fixed top-1000 page pool and reranking moved the gold doc to rank `4`

This was a constrained pool experiment, not a recall improvement.

### 3b29528f6d900ff20bfebd2b938b851f

Question:

- `Which African American artist was a musical guest at the 2003 Sanremo Music Festival?`

Finding:

- query label file marked `visual_token_indices: []`
- not a useful case for the current visual-aware reranker unless labels are overridden

### 3b3793152347552aea6d81cf2d24a82b

Question:

- `Which Incumbent(s), in Present German cabinet of Cabinet of Germany, is a woman with short dark hair?`

Gold doc from dataset:

- `e51e7755477179a20eab8310a1c25559`

Manual concern:

- supporting doc looked noisy / questionable for exact answer grounding

Do not prioritize this qid for clean page-level evaluation.

## Current best transfer config

Best fixed weights found on the strongest page-level success case so far
(`46a4103ba65b176fba9ed85889775f8d`, targeting answer-bearing `page17`):

- `base = 1.0`
- `visual = 1.0`
- `non_visual = 0.0`
- `balance = 8.0`

This is the recommended first fixed-weight transfer setting for the next qid before doing qid-specific grid search.

## Recommended next experiment

Use a fixed-weight transfer run on:

- `3444052221c1104c977a4653988d44f1`

Suggested command:

```bash
python scripts/rerank_target_docs_visual_aware.py \
  --qid 3444052221c1104c977a4653988d44f1 \
  --gold /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl \
  --embedding_name colpali-v1.2_m3-docvqa_dev \
  --query_token_filter drop_pad_like \
  --splice-query-token-labels /mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/visual_needed_binary/deberta_v3_large_seed42/export/dev_query_visual_binary_labels_union_relaxed_v2.jsonl \
  --splice-patch-labels-jsonl /mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/layout_patch_assignments_done_so_far_plus_new_3class_full.jsonl \
  --baseline-pred /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output/retrieval_only_dev_ret1000full_drop_pad_like/colpali-v1.2_ivfflat_ret1000_qtf-drop_pad_like_2026-04-16_07-21-44.json \
  --from-baseline-top-pages 1000 \
  --weight-base 1.0 \
  --weight-visual 1.0 \
  --weight-non-visual 0.0 \
  --weight-balance 8.0 \
  --output-json /mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/34440522_rerank_fixed_best46a4103b.json \
  --output-prediction-json /mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/34440522_rerank_fixed_best46a4103b.pred.json
```

After that, if promising, compare against a qid-specific `--grid-search` run.

## Suggested narrative for future analysis

The reranker story is now:

1. Old patch-label coverage was a real blocker.
2. The new `...plus_new_3class_full.jsonl` file activates meaningful visual/nonvisual channels.
3. With page-level supervision, the helper can substantially improve some true answer-bearing pages.
4. The current linear fusion still struggles with identity-specific discrimination when multiple distractor portrait pages share the same generic cue.

## Update: 2026-04-28

This section supersedes the older "current best transfer config" discussion above.

### Additional helper changes now on `main`

More helper-only commits were added after the initial note:

- `4f6b919` Add repo-local gold-file fallback
- `21ba006` Add batch visual rerank runner
- `c0c263a` Add qid-grid mode to the batch runner
- `00e708f` Enforce offline-only model resolution
- `f8320d2` Add repo/local fallback for embeddings
- `eac3917` Remove stale plotting import from patch-label loading
- `4e35acc` Force helper scripts to prefer local `src/`
- `2741f3c` Add query decomposition rerank helper
- `8697a62` Add retrieval-side token-filter control to decomposition helper
- `49bc706` Fix decomposition helper retrieval call
- `2bf9f07` Set the current preferred defaults

For HPC runs, the stable environment setup is:

```bash
export PYTHONPATH=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/src${PYTHONPATH:+:$PYTHONPATH}
export LOCAL_MODEL_DIR=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/model
export LOCAL_EMBEDDINGS_DIR=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/embeddings
```

### Current preferred default

After the batch comparisons below, the preferred default setting is:

- `query_token_filter = full`
- `weight_base = 1.0`
- `weight_visual = 1.0`
- `weight_non_visual = 0.0`
- `weight_balance = 8.0`

This is now the default in both:

- `scripts/rerank_target_docs_visual_aware.py`
- `scripts/run_visual_rerank_batch.py`

### Batch result on the 83 ImageListQ failures

Input set:

- `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output/rag_dev_ret4/ret4_imagelistq_failures_no_gold_doc_in_top4.jsonl`

#### Fixed config on `drop_pad_like` top-1000 pool

Summary:

- `num_qids = 83`
- `baseline_top4_doc_count = 7`
- `reranked_top4_doc_count = 21`
- `improved_doc_rank_count = 58`
- `worsened_doc_rank_count = 13`
- `unchanged_doc_rank_count = 4`
- `baseline_doc_rank_median = 86.0`
- `reranked_doc_rank_median = 14.0`
- `baseline_page_rank_median = 96.0`
- `reranked_page_rank_median = 18.0`

Output files:

- `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/imagelist_ret4_no_gold_top4_rerank_fixed_balance8.jsonl`
- `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/imagelist_ret4_no_gold_top4_rerank_fixed_balance8.summary.json`

#### Fixed config on `full` top-1000 pool

Summary:

- `num_qids = 83`
- `improved_doc_rank_count = 54`
- `reranked_top4_doc_count = 32`

Output files:

- `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/imagelist_ret4_no_gold_top4_rerank_full_balance8.jsonl`
- `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/imagelist_ret4_no_gold_top4_rerank_full_balance8.summary.json`

#### Head-to-head: `full` vs `drop_pad_like`

Per-qid comparison:

- `full_better_count = 32`
- `drop_better_count = 28`
- `tie_count = 23`
- `both_top4 = 19`
- `full_top4_only = 13`
- `drop_top4_only = 2`

Average gold-doc rank on the overlap where both sides are non-`None`:

- `avg_drop_rank = 56.86`
- `avg_full_rank = 45.96`

Conclusion:

- `full` is the better default overall
- `drop_pad_like` improves slightly more qids in the aggregate count
- but `full` produces many more top-4 wins, which is the more important metric here

### Worsened-13 qids under qid-specific grid search

The 13 qids that got worse under the fixed `drop_pad_like` setting were rerun with qid-specific grid search.

Grid summary:

- `num_qids = 13`
- `improved_doc_rank_count = 3`
- `reranked_top4_doc_count = 0`

Compared against the bad fixed-config result:

- `rescued_vs_fixed = 6`
- `rescued_to_baseline_or_better = 3`
- `top4_after_grid = 0`
- `still_worse_than_baseline = 10`

Interpretation:

- weight tuning helps some of the 13 relative to the fixed bad run
- but most of these failures are not recoverable by weight search alone
- they are likely feature / labeling / candidate-pool problems, not just a bad global weight choice

### LGBT qid update: `e783cba0b3df36372d11823e378e5437`

Question:

- `Which completely bald person who wears thick glasses is among the members of LGBT billionaires?`

Gold doc:

- `d57e56eff064047af5a6ef074a570956`

#### Natural `full` top-1000 pool

Using the natural `ret1000 full` baseline:

- `first_gold_doc_rank = None`
- `n_gold_page_hits = 0`

So this is a recall failure under the default first-stage pool.

#### Forced-page tests under the new default family

Injecting gold pages into the `full` top-1000 pool and reranking with the current default family did not rescue the qid:

- forced `page0` clean rerun:
  - `first_gold_doc_rank = 77`
- qid-specific grid search with forced `page0`:
  - `first_gold_doc_rank = 215`
  - `first_gold_page_rank = 252`
  - best weights:
    - `base = 1.0`
    - `visual = 4.0`
    - `non_visual = 0.0`
    - `balance = 8.0`

Interpretation:

- even when the right page is forced into the pool, the current visual-aware family does not recover this qid

#### Query decomposition helper results

Dense-only decomposition with:

- original full question
- `LGBT billionaires`
- `bald person thick glasses`
- `LGBT billionaire bald glasses`

did not surface the gold doc at all.

Semantic-only retrieval-side decomposition did better. Per-subquery best gold-doc hits:

- original question:
  - rank `502`
- `LGBT billionaires`:
  - rank `37`
- `bald person thick glasses`:
  - rank `311`
- `LGBT billionaire bald glasses`:
  - no hit

The best semantic subquery is clearly:

- `LGBT billionaires`

#### RRF over the four semantic-only subquery pools

With `retrieval_query_token_filter = semantic_only`, `top-pages-per-query = 1000`, and RRF merge:

- `candidate_doc_count = 1620`
- `candidate_page_count = 3481`
- `merged_first_gold_doc_rank = 168`
- `reranked_first_gold_doc_rank = 723`

Interpretation:

- semantic decomposition improves recall
- but the current visual-aware reranker is actively harmful on that merged pool

#### Single-subquery semantic pool: `LGBT billionaires`

Using only the semantic subquery:

- `merged_first_gold_doc_rank = 31`

Then comparing rerankers on the exact same pool:

- current default visual-aware reranker:
  - `reranked_first_gold_doc_rank = 209`
- base-only rerank:
  - `reranked_first_gold_doc_rank = 9`

This is the strongest current result for the LGBT qid.

Interpretation:

- semantic/category retrieval is the right direction for this qid
- once the right semantic pool is found, plain base-only reranking helps a lot
- the current visual-aware extras are catastrophically harmful on that pool

### Practical takeaway

There is now a useful conditional narrative:

1. For broad ImageListQ recovery, use:
   - `query_token_filter = full`
   - `base = 1.0`
   - `visual = 1.0`
   - `non_visual = 0.0`
   - `balance = 8.0`
2. For hard semantic/category-heavy cases like the LGBT qid:
   - semantic subquery retrieval can recover the right doc neighborhood
   - base-only reranking is safer than the current visual-aware fusion
3. The next likely improvement is multi-source retrieval:
   - semantic/category retrieval source
   - dense full retrieval source
   - possibly later sparse/title/entity retrieval
   - followed by conditional reranking rather than one universal fusion

## Update: 2026-04-30

### Additional helper changes now on `main`

New helper-only commits after the 2026-04-28 note:

- `226a2a5` Add baseline-score source option for visual reranker
- `fdafddb` Add query-label version comparison helper
- `1a56a20` Optimize base-only visual reranker path
- `9b2e1ef` Add top-k pruning for base-only MaxSim
- `bef1226` Separate approximate MaxSim base path
- `a4276d8` Fix top-k MaxSim pruning import

These add:

- `--base-score-source baseline_pred`
- `--base-score-source approx_page_maxsim_topk`
- `--approx-base-page-token-topk K`
- `scripts/compare_query_label_versions.py`

The exact path is still preserved when:

- `--base-score-source exact_page_maxsim`

The approximation is now isolated to its own path and does not silently change the exact one.

### Dev-set visual-evidence stats

Using MMQA dev taxonomy:

- total dev questions: `2441`
- total question types: `16`
- image-involving question types: `10 / 16`
- dev questions in image-involving types: `940 / 2441 = 38.51%`

Using the current best query-label export:

- `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/visual_needed_binary/deberta_v3_large_seed42/export/dev_query_visual_binary_labels_union_relaxed_v6_fulltrainlex_v2.jsonl`

Labeler-wide counts:

- questions with at least one visual-needed token/phrase:
  - `999 / 2441 = 40.93%`
- image-type questions labeled visual-needed:
  - `854 / 940 = 90.85%`

Interpretation:

- MMQA question type is the right dataset-level proxy for "requires image evidence"
- `v6_fulltrainlex_v2` is the right labeler-level view of "did we recover an explicit visual cue?"

### Query-label file comparison: `union_relaxed_v2` vs `v6_fulltrainlex_v2`

On sampled qids such as:

- `e783cba0b3df36372d11823e378e5437`
- `3444052221c1104c977a4653988d44f1`
- `46a4103ba65b176fba9ed85889775f8d`
- `e1e6ed53f9ad11813845088f4cf2f6b1`

the raw JSON fields can look different, but the only trustworthy comparison is the final ColPali-aligned query-token class sequence used by the reranker.

On those sampled qids:

- the effective aligned visual-token labels were identical across `v2` and `v6`

On all `141` `ImageListQ` qids, using `scripts/compare_query_label_versions.py` with ColPali-aligned final token classes:

- `total_qids = 141`
- `identical_class_sequence_count = 73`
- `different_class_sequence_count = 68`
- `gained_visual_qid_count = 66`
- `lost_visual_qid_count = 2`

Interpretation:

- `v6` changes the effective reranker input on nearly half of `ImageListQ`
- the shift is almost entirely additive
- `v6` mostly adds visual cues rather than removing them

### 83-qid ImageListQ failure subset: additional ablations

Input set remains:

- `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output/rag_dev_ret4/ret4_imagelistq_failures_no_gold_doc_in_top4.jsonl`

Using qid-specific grid search with the old query-label file (`union_relaxed_v2`) and exact page-local MaxSim:

#### H1: `base + visual + non_visual` with `balance = 0`

- `num_qids = 83`
- `improved_doc_rank_count = 61`
- `reranked_top4_doc_count = 39`

#### H2: `base + visual` with `non_visual = 0`, `balance = 0`

- `num_qids = 83`
- `improved_doc_rank_count = 59`
- `reranked_top4_doc_count = 38`

Interpretation:

- a simpler fused score can carry most of the benefit
- `non_visual` adds only a small extra gain in this oracle-style qid-grid setting
- these are still qid-specific tuned results, not a single deployable global configuration

#### Fixed `v6` run with the old preferred global weights

Using:

- `query_token_filter = full`
- `base = 1.0`
- `visual = 1.0`
- `non_visual = 0.0`
- `balance = 8.0`
- query labels = `v6_fulltrainlex_v2`

Summary:

- `num_qids = 83`
- `improved_doc_rank_count = 53`
- `reranked_top4_doc_count = 30`

Compared with the earlier fixed `v2` run:

- fixed `v2`: `54` improved, `32` top-4
- fixed `v6`: `53` improved, `30` top-4

Interpretation:

- `v6` did not help this subset under the old global weights
- if `v6` is used in reranking, it likely needs retuning rather than inheriting the `v2` fixed weights

### All `141` ImageListQ qids

Using the original fixed full-method run on all `ImageListQ` qids:

- output:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/imagelistq_all_rerank_full_balance8.jsonl`
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/imagelistq_all_rerank_full_balance8.summary.json`

Summary:

- `num_qids = 141`
- `baseline_top4_doc_count = 53`
- `reranked_top4_doc_count = 82`
- `improved_doc_rank_count = 75`
- `worsened_doc_rank_count = 24`
- `unchanged_doc_rank_count = 31`
- `baseline_doc_rank_median = 8.5`
- `reranked_doc_rank_median = 2.0`

The bucket counts above do not sum to `141` because:

- `11` qids had `baseline_first_gold_doc_rank = None`
- and also `reranked_first_gold_doc_rank = None`

So:

- `11 / 141 = 7.80%` are unrecoverable by reranking alone under the fixed top-1000 pool

Clean interpretation:

- baseline top-4 gold-doc rate:
  - `53 / 141 = 37.59%`
- full reranker top-4 gold-doc rate:
  - `82 / 141 = 58.16%`
- gain:
  - `+20.57` percentage points

### All `141` ImageListQ qids: base-only exact MaxSim

Using:

- query labels = `v6_fulltrainlex_v2`
- `base-score-source = exact_page_maxsim`
- `base = 1.0`
- `visual = 0.0`
- `non_visual = 0.0`
- `balance = 0.0`

Output:

- `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/imagelistq_all_rerank_baseonly_exactmaxsim_v6.jsonl`
- `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/imagelistq_all_rerank_baseonly_exactmaxsim_v6.summary.json`

Summary:

- `num_qids = 141`
- `improved_doc_rank_count = 73`
- `reranked_top4_doc_count = 78`

Interpretation:

- baseline top-4 gold-doc rate:
  - `53 / 141 = 37.59%`
- base-only exact-MaxSim top-4 gold-doc rate:
  - `78 / 141 = 55.32%`
- gain:
  - `+17.73` percentage points

Compared with the full reranker on the same `141` qids:

- full reranker:
  - `82 / 141`
- base-only exact MaxSim:
  - `78 / 141`

So the full method is still better, but only by:

- `4` extra top-4 recoveries

Interpretation:

- much of the gain comes from recomputing exact page-local MaxSim on the fixed top-1000 pool
- the visual-aware terms add a smaller but real extra gain

### Approximate base-only MaxSim with query-guided page-token pruning

This was tested on a stress subset:

- `20` qids chosen from cases rescued into top-4 by the normal full reranker

Important caveat:

- this rescued-20 subset was constructed from the older all-141 full reranker output because the all-141 full-`v6` output was not yet available at subset-construction time

Outputs:

- exact base-only:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/imagelistq_rescued_by_full_top20_baseonly_exact.summary.json`
- approximate `top128`:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/imagelistq_rescued_by_full_top20_baseonly_top128.summary.json`
- approximate `top256`:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/imagelistq_rescued_by_full_top20_baseonly_top256.summary.json`
- approximate `top512`:
  - `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/imagelistq_rescued_by_full_top20_baseonly_top512.summary.json`

Summary on the same rescued-20 subset:

- exact base-only:
  - `reranked_top4_doc_count = 12`
  - `reranked_doc_rank_median = 3.0`
- `top128`:
  - `reranked_top4_doc_count = 12`
  - `reranked_doc_rank_median = 3.5`
- `top256`:
  - `reranked_top4_doc_count = 13`
  - `reranked_doc_rank_median = 3.0`
- `top512`:
  - `reranked_top4_doc_count = 10`
  - `reranked_doc_rank_median = 4.0`

All four variants had:

- `improved_doc_rank_count = 19`
- `worsened_doc_rank_count = 0`
- `unchanged_doc_rank_count = 1`

Interpretation:

- query-guided top-K page-token pruning is promising
- `top256` is the best setting seen so far on this subset
- larger `K` is not better here; `top512` is worse than `top256`
- the pruning is acting like a regularizer, not just an approximation

### Practical current view

1. For broad `ImageListQ` retrieval quality, the fixed full reranker is still the strongest global result we have:
   - `82 / 141` top-4 on all `ImageListQ`
2. Base-only exact MaxSim already gives most of that gain:
   - `78 / 141` top-4
3. The difference between full and base-only is real but modest:
   - `4` extra top-4 recoveries on all `ImageListQ`
4. `v6_fulltrainlex_v2` substantially changes label coverage, but fixed old `balance=8` weights do not automatically improve with it
5. For approximate base-only MaxSim:
   - `top256` is the only promising first setting so far
   - it should be tested next on a larger set if a speed/quality tradeoff is needed

## Update: 2026-04-30

This section supersedes the older "82 / 141" broad-score discussion above. The current stable reference runs use:

- query labels:
  - `dev_query_visual_binary_labels_union_relaxed_v6_fulltrainlex_v2.jsonl`
- patch labels:
  - `layout_patch_assignments_done_so_far_plus_new_3class_full.jsonl`
- baseline page pool:
  - `ret1000 full`

### Helper changes now on `main`

Additional helper-only commits added after the previous note:

- `78e1804` Add two-stage base-only MaxSim reranking
- `aa6cb65` Fix page selection in two-stage MaxSim reranking
- `c48aaf4` Add spatially diverse top-k token selection for MaxSim pruning
- `2da9cfc` Fix torch import in spatial token selector
- `fe2c3b8` Add query-label-aware token selection for MaxSim pruning
- `f7010a6` Add base-only batching and coarse dtype controls
- `d57bd10` Add visual rerank run comparison helper
- `3fe8884` Add visual rerank subset filter helper
- `5eb43e4` Add soft label prior for top-k MaxSim pruning
- `f12d8f9` Add staged visual reranking over top pages
- `69e92e2` Add doc-level exact refinement after top-k MaxSim

### Current all-141 ImageListQ scoreboard

Current stable runs on all `141` `ImageListQ` qids:

- baseline control (`baseline_pred` reused as-is):
  - `53 / 141` top-4
  - `0 / 141` improved
- base-only exact page MaxSim:
  - `78 / 141` top-4
- base-only approximate top-K MaxSim:
  - `top256`
  - `query_mean`
  - `global_topk`
  - `fp32`
  - `79 / 141` top-4
  - `74 / 141` improved
- full visual-aware `v6`:
  - `base = 1.0`
  - `visual = 1.0`
  - `non_visual = 0.0`
  - `balance = 8.0`
  - `80 / 141` top-4
  - `75 / 141` improved

Clean interpretation:

- most of the gain comes from recomputing strong page-local base scores
- `top256` is the best low-overhead approximation seen so far
- the full visual-aware method is still the best tested absolute result, but only by `+1` over `top256`

### Exact vs top256 disagreement analysis

Head-to-head on all `141` qids:

- `top4_only_in_exact = 2`
- `top4_only_in_top256 = 3`
- `top4_in_both = 76`
- `top4_in_neither = 60`

So `top256` is very close to exact in practical outcome. It does not behave like a broken approximation; it mainly changes a handful of near-boundary cases.

For the `top256` run:

- unrecoverable under the fixed top-1000 pool:
  - `11` qids with `reranked_first_gold_doc_rank = None`
- near-miss subset:
  - `19` qids with reranked gold-doc rank in `5..20`

This `19`-qid near-miss subset is the most actionable rescue set for later-stage reranking.

### Negative results: methods that did not help

The following variants were tested and are not currently recommended:

- stronger coarse scorer:
  - `query_token_max`
  - matched exact-like behavior but did not beat `top256 + query_mean`
- page-level two-stage exact refinement:
  - `top256 -> exact top50 / top100 / top200 pages`
  - did not beat plain `top256`
- spatially diverse token selection:
  - worse than plain global `top256`
- hard query-label token reserve:
  - worse than plain global `top256`
- coarse `bf16` scoring:
  - catastrophically unstable on all `141` qids
  - `8 / 141` top-4
- doc-level exact refinement on the generic near-miss set:
  - `top256 -> exact full MaxSim on top100 docs`
  - only `2 / 19` top-4 on the near-miss subset

Interpretation:

- many failures are not fixed by "more exact base MaxSim" alone
- global visual-aware reranking also has limited marginal value on the whole pool

### Selective visual reranking: the most promising new direction

The new staged mode:

- first runs cheap `top256` base-only ranking on the whole candidate pool
- then recomputes full visual-aware features only for the top `N` stage-1 pages

CLI:

- `--visual-rerank-top-pages N`

Important result on the `19`-qid near-miss subset:

- `top256 -> staged top50 full_v6`
  - `8 / 19` top-4
  - `14 / 19` improved vs baseline

This is the strongest recent signal for using visual scoring efficiently:

- visual scoring is not a high-leverage global signal
- but it is useful as a selective late reranker on boundary cases

The all-141 staged run was queued after this note update and should be evaluated against:

- base-only `top256`:
  - `79 / 141`
- full global visual-aware:
  - `80 / 141`

If staged top-50 lands near `80 / 141`, it is a much better efficiency story than running the full visual-aware reranker on all `1000` candidate pages.

### Current recommended configurations

Best absolute tested result on all `141`:

- full visual-aware rerank
- `query_token_filter = full`
- `base_score_source = exact_page_maxsim`
- `base = 1.0`
- `visual = 1.0`
- `non_visual = 0.0`
- `balance = 8.0`

Best efficiency / quality tradeoff tested so far:

- base-only approximate MaxSim
- `query_token_filter = full`
- `base_score_source = approx_page_maxsim_topk`
- `approx_base_page_token_topk = 256`
- `approx_base_page_token_scorer = query_mean`
- `approx_base_page_token_selector = global_topk`
- `approx_base_page_token_coarse_dtype = fp32`
- `base = 1.0`
- `visual = 0.0`
- `non_visual = 0.0`
- `balance = 0.0`

Most promising next deployment pattern:

- use `top256` as the global cheap reranker
- then run staged visual reranking only on a small shortlist, especially for the near-miss band (`5..20`)

### Practical current view

1. Strong page-local base rescoring is the main retrieval win.
2. `top256` preserves essentially all of that gain at much lower cost.
3. Visual-aware scoring seldom changes the global outcome, but it can matter on near-boundary logo / poster / appearance queries.
4. Selective late visual reranking is more promising than global heavy visual reranking.
5. Exact doc-level refinement after `top256` is not a strong generic fix; it should only be stress-tested on qids where `top256` is specifically worse than exact.

### Later 2026-04-30 update: full-pool transfer checks

#### Staged visual rerank did not transfer to all `141`

After the promising `19`-qid near-miss result, the staged full-pool run was completed on all `141` `ImageListQ` qids:

- config:
  - stage 1:
    - `base_score_source = approx_page_maxsim_topk`
    - `approx_base_page_token_topk = 256`
    - `approx_base_page_token_scorer = query_mean`
    - `approx_base_page_token_selector = global_topk`
  - stage 2:
    - `visual_rerank_top_pages = 50`
    - `base = 1.0`
    - `visual = 1.0`
    - `non_visual = 0.0`
    - `balance = 8.0`

Result:

- `improved_doc_rank_count = 78`
- `reranked_top4_doc_count = 78`

Interpretation:

- the staged `top50` visual rerank improved many qids in rank terms
- but it did not beat either:
  - base-only `top256`:
    - `79 / 141` top-4
  - full global visual-aware:
    - `80 / 141` top-4
- so the local near-miss success did not transfer to the full `141`

Practical ranking remains:

1. full global visual-aware:
   - `80 / 141` top-4
2. base-only `top256`:
   - `79 / 141` top-4
3. base-only exact:
   - `78 / 141` top-4
4. staged `top50` visual rerank:
   - `78 / 141` top-4

#### Doc-level exact refinement is targeted, not general

The new `two_stage_doc_maxsim` path was evaluated in two ways.

On the generic `19`-qid near-miss subset:

- `top256 -> exact full MaxSim on top100 docs`
  - `2 / 19` top-4

This is weak and confirms that most near-miss failures are not fixed by reverting to exact base MaxSim alone.

On the `34` qids where `top256` was specifically worse than exact:

- `docstage50`
  - `8 / 34` top-4
  - reranked median `17.5`
- `docstage100`
  - `8 / 34` top-4
  - reranked median `16.0`
- `docstage200`
  - `8 / 34` top-4
  - reranked median `16.0`

Interpretation:

- this method has a mild repair effect on the "top256 hurt relative to exact" subset
- its benefit saturates by `top100` docs
- `top200` adds cost without any measured gain
- this is not a strong general second-stage default

If this family is kept at all, `two_stage_doc_maxsim` with:

- `two_stage_exact_top_docs = 100`

is the representative setting.

#### Top-10 view from the existing outputs

Using the saved per-qid `jsonl` outputs, the methods can also be compared at `top10` without rerunning anything.

Top-10 gold-doc counts on all `141` `ImageListQ` qids:

- baseline control:
  - `72 / 141`
- base-only exact:
  - `90 / 141`
- base-only `top256`:
  - `88 / 141`
- full global visual-aware:
  - `90 / 141`
- staged `top50` visual rerank:
  - `87 / 141`

Top-10 gold-page counts:

- baseline control:
  - `65 / 141`
- base-only exact:
  - `84 / 141`
- base-only `top256`:
  - `84 / 141`
- full global visual-aware:
  - `89 / 141`
- staged `top50` visual rerank:
  - `85 / 141`

Interpretation:

- at `top4`, `top256` slightly beats exact (`79` vs `78`)
- at `top10`, exact is better than `top256` for docs (`90` vs `88`)
- full global visual-aware ties exact at `top10 doc`, but is best at `top10 page`

This suggests:

- `top256` is strongest for aggressive early precision / top-4 boundary wins
- exact base rescoring is slightly better for broader top-10 recovery
- visual-aware scoring helps more on page surfacing than on additional doc-level top-10 wins

#### Subset definition reminder

There are two different "visual" subset notions in the project:

1. question-type level:
   - the `10 / 16` MMQA image-involving question types
   - this is the right subset for "requires visual evidence to answer"
2. query-label level:
   - qids whose query-label export contains explicit visual-needed cues
   - this is the right subset for "does the question text expose a visual cue"

Do not mix these in later analysis; they answer different questions.

## Addendum: 2026-05-01

### New helper changes now on `main`

Additional helper-only commits added after the previous note:

- `52acb63` Add hard-gated auxiliary rerank by base doc rank
- `d218059` Add soft base-gated auxiliary reranking

New flags introduced:

- `--gated-visual-top-docs R`
  - only apply non-base channels (`visual`, `non_visual`, `balance`) to docs whose
    stage-1 base-only doc rank is `<= R`
- `--scale-auxiliary-by-base-score`
  - multiply the auxiliary bonus by:
    - `base_page_score / max_base_page_score`
  - intended to let visual evidence help more on already-strong base pages

### Gated / tie-break visual rerank family on the `19` near-miss qids

The following methods were tested on:

- `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/imagelistq_all_top256_nearmiss_rank5to20.jsonl`

Results:

- staged late visual tie-break:
  - `visual_rerank_top_pages = 50`
  - informative-visual-query gating
  - preserve stage-1 base score
  - `2 / 19` top-4
- hard-gated visual rerank:
  - `gated_visual_top_docs = 20`
  - `visual = 0.25`
  - `balance = 1.0`
  - `2 / 19` top-4
- soft-gated visual rerank:
  - `gated_visual_top_docs = 20`
  - `scale_auxiliary_by_base_score = true`
  - `visual = 0.25`
  - `balance = 1.0`
  - `2 / 19` top-4
- soft-gated visual rerank:
  - same as above, but `visual = 0.5`
  - `2 / 19` top-4
- soft-gated visual rerank:
  - same as above, but `visual = 1.0`
  - `5 / 19` top-4
  - `14 / 19` improved vs baseline

Interpretation:

- the gated / tie-break family is real, but weak at low visual weight
- `visual = 1.0` is the first setting in this family that gives a meaningful rescue effect
- even then, it still trails the strongest earlier selective rerank:
  - `top256 -> staged top50 full_v6`
  - `8 / 19` top-4

So the best current ordering on the `19` near-miss set is:

1. staged `top50` full visual rerank:
   - `8 / 19`
2. soft-gated visual rerank with `visual = 1.0`:
   - `5 / 19`
3. doc-stage exact refinement / hard gate / late tie-break:
   - around `2 / 19`
4. pruning-prior variants:
   - `0 / 19`

### Visual-only low-level audits

Pure visual-only runs were used to test whether the visual channel is functioning mechanically:

- weights:
  - `base = 0.0`
  - `visual = 1.0`
  - `non_visual = 0.0`
  - `balance = 0.0`

#### `4fc70c64c8abe430a1af267700e290b8` (`paw-print logo`)

- gold doc:
  - rank `3`
- gold page:
  - rank `3`

Interpretation:

- the visual channel is clearly alive
- it can surface the true evidence page near the top without any base signal

#### `9eaf685adaccf2218cc3d8fcf8797d09` (`shield-shaped logo`)

Global visual-only run:

- gold doc:
  - rank `51`
- gold page `page0`:
  - rank `110`

Doc-only visual audit inside the gold doc:

- `page7`:
  - rank `1`
  - visual score `3.1228`
- `page0`:
  - rank `2`
  - visual score `2.6156`

Manual audit:

- `page0` is the true evidence page with a large shield-shaped logo
- `page7` also contains a small shield logo on a train
- several globally higher-ranked competitor pages also contain real shield-shaped logos

Interpretation:

- the visual channel is concept-sensitive
- but it does not reliably choose the strongest / most useful evidence page
- this is not a hallucination bug; it is a specificity / ranking-granularity problem

#### `39d1230b9456528d49ced799393985d3` (`flaming torch logo`)

Global visual-only run:

- gold doc:
  - rank `9`
- manually chosen gold page `page1`:
  - rank `416`

Doc-only visual audit inside the gold doc:

- `page0`:
  - rank `1`
  - visual score about `3.096`
  - `visual_patch_count = 58`
- `page1`:
  - rank `4`
  - visual score `0.0`
  - `visual_patch_count = 0`

Manual audit:

- both `page0` and `page1` contain the torch logo
- `page0` contains a larger / clearer torch
- several globally higher-ranked competitor pages also contain real torch logos or torch-like logos

Interpretation:

- this qid exposes two separate failure modes:
  1. patch-label miss:
     - real visual evidence can get zero visual score if no patches are labeled `visual`
  2. weak specificity:
     - many competing pages contain the queried visual concept too

### Updated interpretation of the visual signal

These audits support a more precise conclusion:

- visual evidence is real
- the visual channel is mechanically working
- but it is often not strong or specific enough to beat semantically plausible competitors

In other words:

- the system can often detect:
  - `shield logo`
  - `torch logo`
  - `paw-print logo`
- but retrieval still fails when:
  - many competitor pages contain the same visual concept
  - the true evidence page has incomplete patch-label coverage
  - the visual cue is not tightly bound to the relevant non-visual entity context

So the main problem is not:

- "visual score is broken"

It is:

- "visual score is too weak / noisy / weakly bound to semantics to dominate ranking on its own"

### Updated view of why `top256` is hard to beat

The current evidence suggests that `top256` is already close to the best result available from this heuristic family:

- exact base-only:
  - `78 / 141`
- base-only `top256`:
  - `79 / 141`
- full global visual-aware:
  - `80 / 141`

Why the margin is so small:

1. `top256` is already a strong regularizer.
   - It removes many distractor page tokens while preserving enough gold evidence.
2. Many remaining failures are outside the fixed top-1000 pool.
   - those cannot be solved by reranking
3. Exact refinement usually removes the regularization benefit.
4. Visual evidence is useful, but only weakly discriminative unless tied to strong non-visual relevance.
5. Patch-label coverage remains a limiting factor.

The best current high-level interpretation is:

- strong page-local base rescoring is the main retrieval win
- `top256` preserves almost all of that win cheaply
- visual-aware features help on a subset of boundary cases
- but current integration methods have not yet produced a robust global improvement over `top256`

### Practical next-step guidance

If visual-aware retrieval work continues, the most justified next directions are:

1. doc-conditioned page reranking:
   - rerank all pages inside the top semantic docs, not just the global top pages
2. fallback visual scoring that does not require visual patch labels
3. stronger visual-semantic conjunction terms
   - pages should be rewarded for:
     - visual cue presence
     - and strong non-visual relevance to the same query

At this point, more small `top256` heuristic tweaks are unlikely to produce large gains by themselves.
