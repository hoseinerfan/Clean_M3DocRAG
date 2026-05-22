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

## Addendum: 2026-05-10

### Reproducibility reset for `plain_top256`

The old saved artifact:

- `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/imagelistq_all_rerank_baseonly_top256_v6.summary.json`

should now be treated as a legacy reference, not the main reproducible baseline.

Controlled reruns showed:

- old artifact:
  - `79 / 141` top-4
- current code on the old retrieval JSON:
  - `78 / 141` top-4
- current code on the current `nprobe=1` retrieval JSON:
  - `78 / 141` top-4

So the `79 -> 78` difference is not caused by retrieval-pool drift. It is a reranker-side legacy artifact difference concentrated in one boundary qid:

- `4fc70c64c8abe430a1af267700e290b8`
  - old artifact reranked gold-doc rank:
    - `4`
  - current rerun on the same old retrieval pool:
    - `5`

The old artifact could not be reproduced from the committed code states tested in the relevant `top256` development window, so it is best explained by a slightly different local state or invocation at the time the old file was produced.

### Current reproducible `ImageListQ` scoreboard

Reproducible current-code results on all `141` `ImageListQ` qids:

- base-only exact page MaxSim:
  - `78 / 141` top-4
- base-only `top256`, current `nprobe=1` retrieval:
  - `78 / 141` top-4
- base-only `top256`, `nprobe=4` retrieval:
  - `79 / 141` top-4
- base-only `top256`, `nprobe=8` retrieval:
  - `79 / 141` top-4

Practical conclusion:

- the current reproducible standalone baseline should be:
  - base-only `top256`
  - `nprobe = 4`
  - `79 / 141` top-4
- `nprobe=8` gives no extra gain over `nprobe=4`

### Full-dev base-only `top256`

The full MMQA dev set (`2441` qids) was first aggregated from the chunked `mmqa_dev_top256` outputs using the older `nprobe=1` retrieval pool:

- old `nprobe=1` baseline top-4:
  - `2054 / 2441`
  - `84.15%`
- old `nprobe=1` + base-only `top256` top-4:
  - `2256 / 2441`
  - `92.42%`
- gain over old `nprobe=1` baseline:
  - `+202` top-4 qids
  - about `+8.28` points

Old `nprobe=1` totals:

- `improved_doc_rank_count = 592`
- `worsened_doc_rank_count = 197`

The same full-dev chunked run was then repeated with the stronger `nprobe=4` retrieval pool:

- `nprobe=4` baseline top-4:
  - `2193 / 2441`
  - `89.84%`
- `nprobe=4` + base-only `top256` top-4:
  - `2277 / 2441`
  - `93.28%`
- gain over `nprobe=4` baseline:
  - `+84` top-4 qids
  - about `+3.44` points

`nprobe=4` totals:

- `improved_doc_rank_count = 425`
- `worsened_doc_rank_count = 201`

Direct comparison:

- `nprobe=4` baseline vs old `nprobe=1` baseline:
  - `+139` top-4 qids
- `nprobe=4` + `top256` vs old `nprobe=1` + `top256`:
  - `+21` top-4 qids

Interpretation:

- `nprobe=4` is clearly worth using for broad full-dev experiments
- full-dev is much more sensitive to upstream retrieval quality than the `141` `ImageListQ` slice suggested
- current best reproducible broad intrinsic setup is:
  - `nprobe=4` retrieval
  - base-only `top256`

### Gold-doc coverage split: retrieval bottleneck vs reranking bottleneck

Old `nprobe=1` top-1000-page-pool gold-doc coverage on the `141` `ImageListQ` qids:

- average gold-doc coverage:
  - `92.0804%`
- full coverage qids:
  - `129 / 141`
- zero-coverage qids:
  - `11 / 141`
- histogram:
  - `{0: 11, 25: 0, 50: 0, 75: 1, 100: 129}`

On the `19` near-miss qids:

- average gold-doc coverage:
  - `100.0%`
- full coverage qids:
  - `19 / 19`
- zero-coverage qids:
  - `0 / 19`

This cleanly separates the problem into:

1. retrieval failures:
   - `11` qids with zero gold-doc coverage in the fixed pool
2. reranking failures:
   - the `19` near-miss qids are all fully covered and are pure reranking difficulty

### `nprobe` changes retrieval coverage, but only slightly improves top-4

On the `141` `ImageListQ` qids, gold-doc coverage by retrieval pool:

- old `nprobe=1`:
  - average `92.0804%`
  - full coverage `129`
  - zero coverage `11`
- new `nprobe=1`:
  - same as old
- `nprobe=4`:
  - average `96.6903%`
  - full coverage `135`
  - zero coverage `4`
- `nprobe=8`:
  - average `96.6903%`
  - full coverage `135`
  - zero coverage `4`
- `nprobe=16`:
  - average `95.9811%`
  - full coverage `134`
  - zero coverage `5`

Relative to old `nprobe=1`, `nprobe=4` rescued `8` of the old `11` zero-coverage qids.

But the `top256` top-4 impact was small:

- current `nprobe=1`:
  - `78 / 141`
- `nprobe=4`:
  - `79 / 141`
- `nprobe=8`:
  - `79 / 141`

The single strict `nprobe=4` gain over current `nprobe=1` was:

- `c3bcf58405d53cdcc9b034c44541e99f`
  - `nprobe=1` baseline gold-doc rank:
    - `244`
  - `nprobe=1` reranked gold-doc rank:
    - `16`
  - `nprobe=4` baseline gold-doc rank:
    - `53`
  - `nprobe=4` reranked gold-doc rank:
    - `1`

So `nprobe=4` is the best practical retrieval setting, but coverage gains rarely become large top-4 gains under the current reranker.

### Exact-after-pruning is the essential ranking step

The coarse-pre-exact diagnostic was added to isolate how much of `plain_top256` comes from:

1. the coarse pruning score itself
2. the final exact MaxSim on the retained tokens

On all `141` `ImageListQ` qids:

- baseline top-4:
  - `53`
- coarse pre-exact top-4:
  - `36`
- exact-after-pruning top-4:
  - `78`

Additional diagnostics:

- `exact_vs_coarse_improved_doc_rank_count = 86`
- `exact_vs_coarse_worsened_doc_rank_count = 30`
- `exact_vs_coarse_unchanged_doc_rank_count = 14`
- median gold-doc rank:
  - coarse pre-exact:
    - `13.0`
  - exact-after-pruning:
    - `3.0`

Interpretation:

- the coarse score is useful for selecting which page tokens to keep
- but it is a poor global ranking score
- the exact MaxSim step after pruning does almost all of the actual ranking work

### Informative-visual query weighting for `query_mean` is active but ineffective

The new informative-visual weighting branch was checked carefully.

Findings:

- `visqweight_1p0` matches a fresh current-code plain-top256 control exactly
- increasing the weight to:
  - `1.5`
  - `2.0`
  - `3.0`
  did not change end-to-end top-4 results on the `19` near-miss qids
- `18 / 19` near-miss qids do contain informative visual query tokens

Page-level check on one qid (`18d374751a67eab5d836ebc2ac95ce82`) showed:

- informative visual query tokens:
  - `['eagle', 'symbol']`
- `max_abs_score_diff = 0.0307357758`
- `mean_abs_score_diff = 0.0062555741`
- `top256_overlap = 246 / 256`
- `changed_tokens = 10`

So the weighting branch is not broken:

- it changes coarse scores
- it changes the retained top-256 page-token set
- but it still does not improve final ranking on the hard near-miss subset

This branch should currently be treated as:

- mechanically correct
- scientifically negative
- not a strong standalone improvement direction

### New scientific branch: principled MaxSim-preserving token pruning

Commit now on `main`:

- `324fc61` Add MaxSim-greedy token pruning diagnostics

This adds a new selector family intended to turn `plain_top256` from a heuristic into a more paper-oriented method.

New selector:

- `--approx-base-page-token-selector maxsim_greedy`

Idea:

- greedily pick page tokens that maximize retained page-query MaxSim mass, rather than using only `query_mean` coarse ranking

New adaptive mode:

- `--approx-base-page-token-adaptive-k-mode maxsim_mass`

Idea:

- stop greedy selection once shifted MaxSim mass preservation reaches a target
- this turns `K` into an adaptive budget rather than a fixed constant

New diagnostics:

- `--report-pruning-diagnostics`

Recorded per-page and aggregated metrics include:

- selected token count
- candidate token count
- full token count
- active query token count
- exact score loss
- shifted MaxSim mass preservation ratio
- argmax retention ratio
- candidate argmax coverage ratio

This is the current scientifically motivated next branch to evaluate.

The branch is designed to support a stronger paper story:

1. objective-aligned pruning
   - the selector tries to preserve the exact MaxSim objective directly
2. adaptive compute control
   - stop once a target preservation threshold is reached
3. measurable explanation of why pruning works
   - score preservation
   - argmax retention
   - candidate coverage

### Current paper-oriented recommendation

If this line of work is pushed toward a peer-reviewed paper, the best next experimental ladder is:

1. baseline control:
   - current reproducible `top256`
   - `global_topk`
   - `nprobe=4`
2. principled fixed-budget selector:
   - `maxsim_greedy`
   - fixed `K=256`
3. adaptive selector:
   - `maxsim_greedy`
   - `maxsim_mass`
   - target preservation around `0.95`
   - adaptive `K` range like `[128, 384]`

The branch is worth keeping only if one of the following happens:

- it beats current reproducible `79 / 141` top-4
- or it matches `79 / 141` while using materially fewer retained tokens on average

That would provide a real accuracy-efficiency tradeoff story instead of another small heuristic tweak.

## 2026-05-11 Addendum

### Fixed-`K` sweep around `256`: stronger pruning is better on the `141`

Using the current reproducible `nprobe=4` retrieval pool on the `141` `ImageListQ` qids, plain base-only `global_topk` behaved as follows:

- `K=224`
  - `81 / 141`
  - `improved_doc_rank_count = 68`
- `K=240`
  - `79 / 141`
  - `improved_doc_rank_count = 69`
- `K=256`
  - `79 / 141`
  - `improved_doc_rank_count = 67`
- `K=272`
  - `79 / 141`
  - `improved_doc_rank_count = 70`
- `K=288`
  - `78 / 141`
  - `improved_doc_rank_count = 70`

Interpretation:

- `256` is not a local optimum on this slice
- `224` is the best fixed budget seen so far on the `141`
- larger `K` tends to drift back toward exact-MaxSim behavior and loses some early-precision regularization benefit

### QID sensitivity split: most unstable qids prefer tighter pruning

The `224..288` sweep was bucketed by qid-level rank trajectories:

- `49` qids:
  - `tight_pruning_prefers`
- `17` qids:
  - `loose_pruning_prefers`
- `75` qids:
  - stable

This was not random noise. Many qids moved smoothly as `K` changed.

The main scientific interpretation is:

- one global `K` is a compromise
- many qids with compact, iconic visual evidence prefer smaller budgets
- a smaller set of more diffuse/contextual qids prefer larger budgets

This supports the adaptive-budget motivation in principle, but the concrete adaptive rules below did not work well enough.

### Adaptive `K`: entropy collapsed, concentration moved but did not separate useful qids

#### `coarse_entropy`

The old `coarse_entropy` adaptive mode was effectively degenerate.

On a probe subset it behaved like:

- fixed `K=256`
  - top-4 `12`
  - mean selected `256`
- adaptive entropy
  - top-4 `12`
  - mean selected `384`

with identical preservation / argmax-retention metrics.

On the full `141`, entropy-adaptive runs also collapsed to the upper cap:

- `224..272`
  - mean selected `272.0`
- `224..288`
  - mean selected `288.0`

So `coarse_entropy` should be treated as a negative result for this use case.

#### `coarse_concentration`

A new adaptive mode was added:

- `9c84b00` Add concentration-based adaptive token budget

This mode was mechanically non-degenerate:

- adaptive concentration `224..272`
  - `79 / 141`
  - `improved_doc_rank_count = 68`
  - mean selected `247.14`
  - median selected `247.0`

Additional ranking-focused diagnostics were added in:

- `dfea652` Add ranking-focused pruning diagnostics

Those diagnostics showed that the adaptive rule still behaved like a near-fixed budget:

- mean top-1 selected token count:
  - `246.80`
- mean selected token count on top-10 pages:
  - `246.90`
- mean selected token count on best pages of top-4 docs:
  - `246.89`
- mean first-gold-page selected token count:
  - `247.03`

Bucketed by the fixed-`K` qid sensitivity split:

- top-1 selected count
  - tight `246.73`
  - loose `246.94`
  - stable `246.81`
- top-4-doc-best mean selected count
  - tight `246.92`
  - loose `246.96`
  - stable `246.85`

So the concentration rule moved the global mean, but it did not meaningfully separate easy vs hard pages at the places that matter for ranking.

Current recommendation:

- do not keep smooth `coarse_entropy` / `coarse_concentration` adaptive rules as the main method
- if adaptive `K` is revisited, it should probably use sharper thresholded bins rather than a weak smooth interpolation

### `maxsim_greedy`: scientifically useful, but still too heavy as a practical method

The candidate-first + lazy-greedy branch remains useful as an interpretability/control line, but it is not a strong practical method.

Recent timed run:

- `maxsim_greedy`
  - `K=256`
  - candidate budget `1024`
  - current lazy-greedy fast path
- result:
  - `78 / 141`
  - `improved_doc_rank_count = 68`
  - wall-clock `4:25:41`
  - max RSS `7.76e6 kB` (`~7.4 GiB`)

Interpretation:

- the branch is still too heavy for a compelling cheap alternative
- and it is not winning enough on quality
- keep it as a scientific control / analysis selector, not the primary method

### New branch: visual-prefiltered exact page rerank

Added in:

- `5122db2` Add visual-prefiltered exact page rerank mode

Method:

1. start from the saved top-1000 baseline page pool
2. compute approximate base-only page scores
3. keep the top `B` pages by base score
4. compute visual-grounded page scores on those pages
5. keep the top `P` pages by the visual prefilter
6. recompute exact MaxSim only on those `P` pages
7. final ranking remains base-only

Important clarification:

- visual / balance scores are used only for page selection into the exact stage
- final fused page score in the reported runs remained:
  - base weight `1.0`
  - visual / non-visual / balance weights `0.0`
- so the final page score is still just base score after the exact refresh

Default scientific config used first:

- base token budget:
  - `224`
- visual shortlist:
  - `B=512`
- exact page budget:
  - `P=128`
- balance mode:
  - `visual_x_grounded_nonvisual_avg`
- grounded context radius:
  - `2`

Result on the `141`:

- `visual_b512_p128_top224`
  - `79 / 141`
  - `improved_doc_rank_count = 72`

Comparison:

- fixed `224`
  - `81 / 141`
  - `improved = 68`
- fixed `256`
  - `79 / 141`
  - `improved = 67`
- visual prefilter exact `B=512, P=128`
  - `79 / 141`
  - `improved = 72`

Interpretation:

- this branch is not dead
- it matches fixed `256` on top-4
- it changes more qids in a favorable direction than fixed `256`
- but `P=128` did not recover the stronger `fixed_224 = 81 / 141` boundary performance

The most reasonable next single ablation is:

- keep `B=512`
- raise exact page budget from `P=128` to `P=192`

### New trace outputs for the visual-prefilter exact branch

Added in:

- `efce4e5` Add visual prefilter page trace outputs

For `visual_prefilter_exact_page_maxsim`, the per-qid JSONL now records:

- `visual_prefilter_exact_page_trace.pre_exact_selected_pages`
  - the selected pages in visual-prefilter order before exact MaxSim
- `visual_prefilter_exact_page_trace.post_exact_selected_pages_by_exact_score`
  - the same selected pages ranked by exact base score
- `visual_prefilter_exact_page_trace.post_exact_selected_pages_final_rerank`
  - those same pages in their final overall page ranking

This makes it possible to inspect exactly which pages entered the exact stage and how exact MaxSim reordered them.

### New helper: visual-prefilter gold-doc coverage on the saved top-1000 baseline pool

Added in:

- `2db27c5` Add visual prefilter coverage helper
- `805d07a` Fix helper import path bootstrap

New script:

- `scripts/check_visual_prefilter_gold_coverage.py`

What it does:

- reuses the ready saved `--baseline-pred` top-1000 page pool
- computes visual-prefilter scores on that fixed pool
- checks gold-doc coverage among the top `N` visually prefiltered pages
- also reports the baseline top-`N` coverage for comparison

It writes:

- per-qid JSONL:
  - baseline top-`N` coverage summary
  - visual-prefilter top-`N` coverage summary
  - top-`N` visual-prefilter page trace
- aggregate summary JSON:
  - mean coverage
  - zero/full-coverage qid counts
  - coverage-improved / worsened / unchanged qid counts

This helper is the right way to test whether visual-prefiltering is actually rescuing or losing gold-doc coverage before spending more time on exact reranking budgets.

### Current recommendation after these experiments

The branch priorities are now:

1. `plain_top224`
   - best fixed-budget intrinsic result on the `141`
2. visual-prefilter exact page rerank
   - most promising new method branch
   - next test should be `B=512, P=192`
3. `maxsim_greedy`
   - keep as interpretability / control, not as the main practical method
4. smooth adaptive `K`
   - not recommended in current form

### Follow-up results: visual-prefilter exact budgets and hard top-50 coverage

Additional exact-page budget sweeps were run for the visual-prefilter branch:

- `visual_b512_p20_top224`
  - `79 / 141`
  - `improved_doc_rank_count = 67`
- `visual_b512_p128_top224`
  - `79 / 141`
  - `improved_doc_rank_count = 72`
- `visual_b512_p192_top224`
  - `78 / 141`
  - `improved_doc_rank_count = 70`

Interpretation:

- `P=20` is surprisingly strong and matches `P=128` on top-4
- `P=128` is still the strongest variant of this branch if total favorable rank movement is the objective
- `P=192` is worse and should not be the default continuation
- none of these variants beat fixed `top224 = 81 / 141`

Hard top-50 coverage on the saved top-1000 baseline pool was also checked:

- baseline mean gold-doc coverage@50
  - `0.696217`
- visual-prefilter mean gold-doc coverage@50
  - `0.380024`

Interpretation:

- the current visual prefilter is too recall-destructive as a hard top-50 filter
- this explains why tiny exact budgets should be treated cautiously if the selected subset is intended to preserve gold-doc coverage directly
- it also suggests the visual signal is better viewed as a ranking perturbation / rescue signal than as a safe hard shortlist by itself

### Reliability audit of visual query tokens and visual page patches

New helper added:

- `743c932` Add visual prefilter qid audit helper

Reliability checklist extension:

- `667cb6c` Add visual prefilter reliability checklist

The audit helper was used on visually explicit qids such as:

- `10b619...`
  - torch on poster
- `39d123...`
  - flaming torch on logo
- `299e684...`
  - gorilla on poster

Representative findings:

- `10b619...`
  - baseline first gold rank `776`
  - visual-prefilter gold rank `280`
  - verdict `useful_but_not_reliable`
- `39d123...`
  - baseline first gold rank `44`
  - visual-prefilter gold rank `68`
  - verdict `borderline`
- `299e684...`
  - baseline first gold rank `4`
  - visual-prefilter gold rank `74`
  - verdict `weak`

Interpretation:

- visual query tokens / visual page patches are informative in a coarse rescue sense
- but they are not reliable enough for aggressive top-50 hard prefiltering
- generic / noisy query-side visual tokens such as `poster`, `logo`, and even `scoring` are a real problem
- grounded support is not consistently strong enough to separate gold pages from visually similar distractors

### Whole-query-to-visual-patches test: negative result

New mode added:

- `513b7c0` Add whole-query visual prefilter score mode

This introduced:

- `visual_query_only`
  - original behavior: visual query tokens vs visual page patches
- `all_query_to_visual_patches`
  - all active query tokens vs visual page patches

On the same 3-qid audit, the whole-query mode did not fix the problem:

- `10b619...`
  - worse: `280 -> 294`
- `39d123...`
  - worse: `68 -> 82`
- `299e684...`
  - slightly better: `74 -> 61`, but still outside top-50

Interpretation:

- query-side visual-label noise is not the only bottleneck
- whole-query scoring raises many pages together but does not improve discrimination enough
- the main failure mode still appears to be poor page-side selectivity among visual distractors

### Prefilter sort-mode comparison on 3 qids: `balance_then_visual` is likely the wrong default

Sort-mode comparisons were added in:

- `a701348` Add visual prefilter sort-mode comparisons

Running the same 3 audited qids with five sort modes gave:

- `balance_then_visual`
  - `10b619...` useful-but-not-reliable, gold rank `280`
  - `39d123...` borderline, gold rank `68`
  - `299e684...` weak, gold rank `74`
- `balance_only`
  - `10b619...` useful-but-not-reliable, gold rank `223`
  - `39d123...` strong, gold rank `2`
  - `299e684...` borderline, gold rank `73`
- `visual_only`
  - `10b619...` useful-but-not-reliable, gold rank `211`
  - `39d123...` strong, gold rank `10`
  - `299e684...` weak, gold rank `190`
- `non_visual_only`
  - `10b619...` useful-but-not-reliable, gold rank `452`
  - `39d123...` strong, gold rank `5`
  - `299e684...` strong, gold rank `5`
- `grounded_non_visual_only`
  - `10b619...` useful-but-not-reliable, gold rank `293`
  - `39d123...` strong, gold rank `21`
  - `299e684...` strong, gold rank `46`

Interpretation:

- `balance_then_visual` likely behaves too much like a visual fallback sorter
- for at least some qids, `balance_only` is dramatically better than `balance_then_visual`
- `visual_only` can rescue some pages but is unstable and can be catastrophic (`299e684...`)
- non-visual / grounded non-visual context can be surprisingly strong in some cases

The likely design flaw is not missing visual tokens or missing visual patches. It is the ranking rule:

- if `balance_score <= 0`, `balance_then_visual` backs off to raw `visual_page_score`
- that allows many visually similar distractors to remain competitive even when conjunction / grounding is weak

### New local confirmation score and explicit page inspection helper

New scorer/tooling added:

- `d013f4b` Add confirmed visual prefilter scoring tools
- `3b17975` Fix inspection helper retrieval model init

Additions:

- a new prefilter score:
  - `confirmed_visual_page_score`
- a new sort mode:
  - `confirmed_visual_only`
- new script:
  - `scripts/inspect_visual_prefilter_pages.py`

The confirmed-visual score uses the current visual anchors, then rescales the visual score by local context support around those anchors rather than trusting page-wide raw visual evidence alone.

The explicit inspection helper allows:

- one qid
- explicit gold page uid(s)
- explicit wrong page uid(s)
- comparison across multiple prefilter modes

and reports:

- full-pool ranks of those chosen pages under each mode
- raw scores:
  - `visual_page_score`
  - `confirmed_visual_page_score`
  - `grounded_non_visual_page_score`
  - `grounded_context_page_score`
  - `non_visual_page_score`
  - `balance_score`
- whether the best gold page beats the chosen wrong page(s)

This helper is the most direct way to test whether a proposed prefilter score can:

- reward the intended gold page
- penalize a visually tempting distractor

before spending more time on full reranking experiments.

### Updated practical recommendation

Current priorities after all follow-up experiments are:

1. `plain_top224`
   - still the strongest fixed-budget base-only result on the `141`
2. `visual_prefilter_exact_page_maxsim`
   - still scientifically interesting
   - but the current prefilter should not be trusted as a hard small shortlist
3. `balance_only`, `non_visual_only`, `grounded_non_visual_only`, and especially `confirmed_visual_only`
   - these are now the most important prefilter scoring directions to audit
4. `maxsim_greedy`
   - keep as interpretability / scientific control
5. smooth adaptive `K`
   - deprioritized

### Whole-page non-visual complement and single-qid workbench

Additional follow-up commits:

- `a4115a6` Add whole-page non-visual complement mode
- `2391dce` Fix helper calls to shared prefilter sort functions
- `eec483e` Add confirmed-visual-gated non-visual prefilter mode

The whole-page non-visual complement mode added:

- `--non-visual-page-mode all_non_visual_patches`

Meaning:

- use all spatial page patches except those labeled visual
- do not rely only on explicitly labeled `non_visual` patches

This was tested first on the earlier hard 3-qid set with:

- `balance_only`
- `balance_score_mode=visual_x_nonvisual_avg`

Result:

- mixed
- `10b619...` improved
- `39d123...` slightly worsened
- `299e684...` worsened

Interpretation:

- page-side non-visual patch-label cleanliness is a real partial bottleneck
- but replacing labeled non-visual patches with the full non-visual complement is too blunt
- this is not the main fix for the hard qids

### New single-qid debugging strategy: pick an exact-MaxSim success case

Instead of continuing to average across hard qids, the next step was to choose one qid that exact MaxSim already solves and use it as a workbench.

Chosen qid:

- `095fc3a41d77542e31001da8c68fd350`
- `There is a flag with a roaring lion on it for which team that is included among the stadiums and locations of the 2017-18 I liga?`

From the exact-MaxSim batch file:

- baseline gold-doc rank: `25`
- exact-MaxSim gold-doc rank: `2`

This makes it a good qid for debugging lightweight prefilters, because exact late interaction clearly has the right signal even though baseline document ranking is not already top-4.

### Key finding on `095fc3...`: this qid is non-visual-context dominant

First qid-level audit with the original helper showed:

- `non_visual_only`
  - strong
  - gold page rank `7`
- `balance_only`
  - borderline
  - gold page rank `70`
- `visual_only`
  - weak
  - gold page rank `134`
- `grounded_non_visual_only`
  - weak
  - gold page rank `136`
- `confirmed_visual_only`
  - weak
  - gold page rank `124`

Interpretation:

- the explicit visual cue itself is not the dominant useful signal here
- page-wide non-visual context is much stronger than the visual-anchor family
- exact MaxSim likely succeeds because it can use full-page interaction rather than a local visual-first prefilter

### Explicit gold-vs-wrong-page inspection on `095fc3...`

Gold page identified in the pool:

- `c8d8e51c58e15c31cd0c91e1ca6e6cd5_page0`

Top wrong pages under `non_visual_only` included:

- `a039a7698902eef86a5a335a22eefab9_page0`
- `a039a7698902eef86a5a335a22eefab9_page2`
- `592b5e2035c06c16997dbc926b174cd7_page12`

Inspection using:

- `non_visual_only`
- `balance_only`
- `grounded_non_visual_only`
- `confirmed_visual_only`
- `balance_score_mode=visual_x_nonvisual_avg`

gave:

- `non_visual_only`
  - `best_gold_rank = 7`
  - `best_wrong_rank = 1`
  - gold loses to wrong pages
- `balance_only`
  - `best_gold_rank = 35`
  - `best_wrong_rank = 186`
  - gold beats the chosen wrong pages
- `grounded_non_visual_only`
  - `best_gold_rank = 136`
  - `best_wrong_rank = 50`
  - gold loses
- `confirmed_visual_only`
  - `best_gold_rank = 124`
  - `best_wrong_rank = 341`
  - gold beats the chosen wrong pages, but rank is still poor

The inspected page scores explain the behavior:

- gold page
  - `visual = 0.4502`
  - `confirmed_visual = 0.1514`
  - `non_visual = 25.0403`
  - `balance = 0.2399`
- wrong `a039...page0`
  - `visual = 0.2546`
  - `confirmed_visual = 0.0677`
  - `non_visual = 28.8160`
  - `balance = 0.1561`
- wrong `a039...page2`
  - `visual = 0`
  - `confirmed_visual = 0`
  - `non_visual = 25.9607`
  - `balance = 0`
- wrong `592b...page12`
  - `visual = 0`
  - `confirmed_visual = 0`
  - `non_visual = 25.6034`
  - `balance = 0`

Interpretation:

- `non_visual_only` finds the gold page early, but also promotes textually strong distractors
- `balance_only` using `visual_x_nonvisual_avg` successfully suppresses visually unsupported distractors
- but multiplicative balance is still too harsh on the gold page itself, because its visual score is only moderate compared with many false positives

This means the right idea on this qid is:

- visual should be a gate
- non-visual should do the ranking

not:

- visual and non-visual should contribute symmetrically through a product-like balance score

### New prefilter mode: `non_visual_with_confirmed_visual_gate`

To reflect the `095fc3...` diagnosis, a new prefilter mode was added in:

- `eec483e` Add confirmed-visual-gated non-visual prefilter mode

New mode:

- `non_visual_with_confirmed_visual_gate`

New flag:

- `--confirmed-visual-gate-threshold`

Behavior:

- if `confirmed_visual_page_score < threshold`, the page is demoted behind all gated-in pages
- among gated-in pages, sort by `non_visual_page_score`

This is intended to:

- keep the strong page-wide semantic ranking of `non_visual_only`
- while removing pages that have no meaningful local visual confirmation

The recommended first sweep on `095fc3...` is:

- `threshold = 0.05`
- `threshold = 0.10`
- `threshold = 0.15`

Current working hypothesis:

- this asymmetric gate-then-rank design is more promising than additional balance formulas
- especially for qids where the gold page has strong non-visual evidence but only moderate visual confidence

## 2026-05-12 Addendum

### Current all-141 top-level ranking

On the `141` `ImageListQ` qids, the current ranking is:

1. best final top-4 result seen so far:
   - `plain_top224`
   - `81 / 141`
2. best current `top256` family results:
   - `plain_top256 mean_uniform`
   - `plain_top256 coverage_topk`
   - both `79 / 141`
3. the recent visual-prefilter families did not beat the best non-visual / plain top-k baselines on the main final metric.

So the strongest final answer-bearing method remains:

- tighter plain pruning at `K=224`

not any of the newer visual-gated or multi-run-union ideas.

### `coverage_topk` vs plain `query_mean`

The `coverage_topk` selector was tested as a more interpretable `top256` variant:

- keep a global chunk of high-scoring page tokens
- reserve part of the `256` budget to cover more distinct informative query tokens

Result on all `141` qids:

- `mean_uniform`
  - `79 / 141`
  - doc-rank median `3`
  - page-rank median `5`
  - mean exact-score loss `1.1239`
  - mean shifted-score preservation `0.9567`
  - mean argmax retention `0.8529`
- `coverage_topk`
  - `79 / 141`
  - doc-rank median `3`
  - page-rank median `5`
  - mean exact-score loss `0.6933`
  - mean shifted-score preservation `0.9730`
  - mean argmax retention `0.9618`

Interpretation:

- `coverage_topk` is clearly a better approximation to exact MaxSim under the same `top256` budget
- but this did not translate into a better final top-4 gold-doc count on the `141`

So `coverage_topk` is currently best viewed as:

- the strongest paper-oriented selector inside the `top256` family
- but not yet a better final reranker than plain `query_mean`

### Multi-gold union idea: negative at matched budget

The idea of unioning multiple top-4 lists from different fixed global variants was tested on the `14` qids where:

- the reference `plain_top256` run had at least one gold doc in top-4
- but not all gold docs in top-4

Matched-budget comparison:

- single `top8` from one run:
  - `6 / 14` full multi-gold coverage
- best union of `2 x top4`:
  - `1 / 14`
- union-only recoveries over single `top8`:
  - `0`

Earlier broader tests already showed:

- single `top12` shortlist was also stronger than union-of-`3 x top4`

Interpretation:

- the union-of-top4 idea should be dropped
- if a longer shortlist is allowed, using one stronger single run is better than unioning several short lists

### Nonspatial prefix/suffix token ablation

The plain `top256` path now supports:

- `--approx-base-page-token-nonspatial-policy keep`
- `--approx-base-page-token-nonspatial-policy drop_all`
- `--approx-base-page-token-nonspatial-policy force_include_all`

on the `global_topk` selector.

All three tied on the main final metric:

- `keep`
  - `79 / 141`
- `drop_all`
  - `79 / 141`
- `force_include_all`
  - `79 / 141`

But their approximation diagnostics differed a lot:

- `drop_all`
  - mean exact-score loss `1.2278`
  - mean shifted-score preservation `0.9523`
  - mean argmax retention `0.8461`
- `keep`
  - mean exact-score loss `1.1239`
  - mean shifted-score preservation `0.9567`
  - mean argmax retention `0.8529`
- `force_include_all`
  - mean exact-score loss `0.1726`
  - mean shifted-score preservation `0.9936`
  - mean argmax retention `0.9004`

Interpretation:

- nonspatial prefix/suffix tokens are not useless noise
- dropping them hurts approximation quality
- forcing them in makes `top256` much closer to exact MaxSim scoring
- but even that stronger fidelity still did not improve the final `79 / 141` top-4 result

This suggests that part of why `plain_top256` works is:

- nonspatial tokens carry useful exact-score signal
- but final top-4 success is still governed by a small set of near-boundary qids rather than raw score-preservation alone

### Current recommendation

If the goal is the strongest current final result:

- use `plain_top224`

If the goal is the cleanest current `top256` selector for a paper-style explanation:

- use `coverage_topk` as the best principled selector
- mention that it improves exact-MaxSim preservation diagnostics
- but note honestly that it did not improve the final `79 / 141` top-4 count on this slice

If the goal is understanding the role of nonspatial tokens:

- the evidence currently says they should not be discarded
- `force_include_all` is the strongest approximation variant among the tested nonspatial policies

## 2026-05-14 Addendum

### Exact-budget sweep completed

The `ImageListQ` exact-budget sweep launched through:

- `c5859db` Add ImageListQ exact-budget sweep SLURM job
- `slurm/imagelistq_exact_budget_sweep.sh`

is now complete on all `141` `ImageListQ` qids.

Final `reranked_top4_doc_count` values:

- baseline exact rerank
  - top5: `61 / 141`
  - top10: `69 / 141`
  - top20: `73 / 141`
  - top50: `77 / 141`
- `nonvisual` prefilter exact rerank
  - top5: `76 / 141`
  - top10: `76 / 141`
  - top20: `76 / 141`
  - top50: `78 / 141`
- `gate005` prefilter exact rerank
  - top5: `77 / 141`
  - top10: `77 / 141`
  - top20: `75 / 141`
  - top50: `74 / 141`

Interpretation:

- `gate005` is the strongest tiny-budget exact rerank variant
  - best at top5
  - tied-best at top10
- `nonvisual` is the most stable prefilter family result
  - flat through top20
  - best prefilter result at top50
- none of these beat `plain_top224`

Current overall ranking on the `141` `ImageListQ` qids:

1. `plain_top224`
   - `81 / 141`
2. `plain_top256_learned_exact_winner`
   - `80 / 141`
3. exact full-page MaxSim
   - `78 / 141`
4. best prefilter exact rerank
   - `nonvisual exact top50`
   - `78 / 141`
5. best tiny-budget prefilter exact rerank
   - `gate005 exact top5/top10`
   - `77 / 141`

So the main headline did not change:

- the strongest proven overall method is still `plain_top224`

### Patch-guided cue-token verifier

Several new helpers were added on `codex/mmdocir-hpc-workflow`:

- `17870be` Add patch-guided crop audit mode
- `d3b0ca4` Add cue-token crop audit scoring
- `6e6da3e` Add batched cue-token crop verifier driver
- `3471524` Add cue-verifier subset builder

The key new idea is:

1. shortlist pages or docs with the existing retriever
2. restrict crop search to visual-patch regions
3. score only cue-specific query tokens such as `dolphin`, `arrow`, or `torch`

This is not a new global reranker yet. It is a routed local verifier for recoverable cases.

#### Clean single-qid results

Distinctive-cue cases behaved well:

- `ef3b89e2bd5909fb1fc7ed65652aea5b`
  - cue: `dolphin`
  - gold page cue-only best crop: `0.8077`
  - strong false positive cue-only best crop: `0.4523`
  - verdict: success
- `9bb02423d6e36d42259f76b72091fce4`
  - cue: `arrow`
  - gold page cue-only best crop: `0.8094`
  - chosen wrong page cue-only best crop: `0.2441`
  - verdict: success
- `39d1230b9456528d49ced799393985d3`
  - cue: `torch`
  - gold page cue-only best crop: `0.5380`
  - chosen wrong page cue-only best crop: `0.2664`
  - verdict: success

Generic-cue cases remained weak:

- `d784b7d93b4bf4d947e512769bbbde25`
  - cue: `soccer + ball`
  - a false positive page with a sports-photo arm patch containing a ball-like badge still scored highly
  - verdict: cue too generic for this verifier form

Interpretation:

- the verifier works best when the cue is distinctive and localized
- it is much less reliable when the cue is generic or requires relation-level reasoning such as:
  - ball inside a team logo
  - not merely any ball-like sports mark on the page

### Reviewed cue-verifier subset

For the `ImageListQ` failure slice with:

- `baseline_first_gold_doc_rank > 4`
- `baseline_first_gold_doc_rank <= 20`

the first reviewed verifier subset was:

- `454a726486b3c8f44571833938ca7cf1`
  - cue `beard`
- `d4d6487894e25da9ba73d0ffb39385b1`
  - cue `orange stripe`
- `2d72990e471477b01f909752e70127ab`
  - cue `stethoscope`
- `e9047078de74d8240ce689bdc504aea7`
  - cue `golden sphere`
- `ee00f4699ce90964a3115771b3e3ba73`
  - cue `motorcycle`
- `9bb02423d6e36d42259f76b72091fce4`
  - cue `arrow`
- `ef3b89e2bd5909fb1fc7ed65652aea5b`
  - cue `dolphin`
- `72d36fc3a4996a9969dc0348167414f1`
  - cue `weasel`
- `29a94164b22263351cc79f515a5b8e8b`
  - cue `lighthouse`

Excluded or deprioritized:

- `beed14e1f1e7d95b34a03d2d152d7424`
  - cue missing from the gold doc during manual audit
- `d784b7d93b4bf4d947e512769bbbde25`
  - cue too generic
- several low-specificity appearance / counting cases

### Current recommendation

If the goal is the strongest overall result:

- use `plain_top224`

If the goal is efficient exact reranking under tiny exact budgets:

- use `gate005` for top5 / top10 budgets
- use `nonvisual` if larger exact budgets are allowed

If the goal is a new routed recovery direction:

- use the cue-token patch-guided verifier only on a manually or heuristically routed subset
- prioritize distinctive cues such as:
  - `dolphin`
  - `arrow`
  - `torch`
  - `weasel`
  - `stethoscope`
  - `motorcycle`

Do not yet claim this verifier as a better global reranker than `plain_top224`.

### Outstanding batch item

The train retrieval-only top-1000 generation job launched through:

- `slurm/m3docrag_dev_pipeline.sh`

failed once because `LOCAL_DATA_DIR` still pointed at the wrong repo tree:

- `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/MMDocIR_M3DocRAG/data/m3-docvqa`

A resubmission with:

- `LOCAL_DATA_DIR=${REPO_ROOT}/data`

was queued afterward. This job is needed before running the train-to-dev learned-selector experiment cleanly.

## 2026-05-21 Addendum

### Retrieval-improver branch

The work since the earlier visual-reranker experiments moved in a different direction:

- keep dense ColPali retrieval
- add a sparse lexical branch with SPLADE
- fuse the two branches at the document level

This work lives on:

- branch `codex/mmdocir-hpc-workflow`

Key commits:

- `83c42dd` Add M3DocVQA page-text export script
- `ff4c055` Add SPLADE page retrieval experiments
- `e1a9e4a` Add routed dense-sparse shortlist evaluator
- `b124d50` Fix routed shortlist metric caps
- `cd88cc3` Add doc-level RRF fusion mode

Relevant scripts:

- `scripts/export_m3docvqa_page_text.py`
- `scripts/build_splade_page_index.py`
- `scripts/run_splade_page_retrieval.py`
- `scripts/fuse_page_retrieval_predictions.py`
- `scripts/route_page_retrieval_predictions.py`
- `mmdocir/analyze_retrieval_by_question_type.py`

### Important methodological split

There are now three different dense+sparse combination ideas in the repo:

1. static dense+sparse tail-union
   - example: `dense7 + splade8 -> top15`
2. heuristic routed shortlist selection
   - `scripts/route_page_retrieval_predictions.py`
   - currently only built-in route mode: `imagelistq_v1`
3. non-heuristic doc-level RRF fusion
   - `scripts/fuse_page_retrieval_predictions.py --fusion-mode doc_rrf`

Current recommendation:

- for a general retrieval improver, use the non-heuristic doc-level RRF fusion
- treat the heuristic router as an `ImageListQ`-specific analysis result, not the main portable method

### MMQA dev page-text and SPLADE index status

Full M3DocVQA/MMQA dev page-text export succeeded with:

- docs: `3366`
- pages: `44294`
- empty-text pages: `78`
- mean char count: `2380.38`
- mean token count: `394.70`

The corresponding SPLADE index build succeeded with:

- page count: `44294`
- stored posting count: `5640379`
- mean terms per page: `127.34`
- zero-term pages: `0`

The full-dev SPLADE retrieval run succeeded with:

- qids: `2441`
- `reranked_top4_doc_count: 2329`
- `reranked_top20_doc_count: 2391`

Important cache setup on HPC:

```bash
export HF_HOME=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/hf_cache
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export HF_DATASETS_CACHE=${HF_HOME}/datasets
```

### ImageListQ-only findings

Corrected `doc@20` on all `141` `ImageListQ` qids:

- `dense_top20`
  - `101 / 141`
- `splade_top20`
  - `105 / 141`
- static hybrid `dense7_splade8_top15`
  - `109 / 141`
- dense+sparse top20 oracle union
  - `116 / 141`

Interpretation:

- the best static non-heuristic tail-union hybrid recovered `11` of the `15` dense+sparse oracle qids
- net gain over dense was `+8`
- there was still a `7`-qid gap to the dense+sparse union oracle

#### Heuristic routed ImageListQ result

Using:

- `scripts/route_page_retrieval_predictions.py`
- `--route-mode imagelistq_v1`
- profile caps:
  - dense `20`
  - sparse `20`
  - hybrid `15`

Result on `ImageListQ`:

- baseline dense top20:
  - `101 / 141`
- routed result:
  - `116 / 141`
- recovered vs dense:
  - `15`
- lost vs dense:
  - `0`

Interpretation:

- the heuristic router reaches the current dense+sparse `ImageListQ` doc-level oracle
- but it does so with handwritten substring rules
- this is useful evidence that routing works
- it is not the preferred portable method for other datasets

### Full MMQA dev retrieval results

Corrected overall full-dev document hit counts:

- `dense_top20`
  - `doc@4 = 2277 / 2441`
  - `doc@20 = 2368 / 2441`
- `splade_top20`
  - `doc@4 = 2329 / 2441`
  - `doc@20 = 2391 / 2441`
- static hybrid `dense7_splade8_top15`
  - `doc@4 = 2277 / 2441`
  - `doc@20 = 2397 / 2441`
- `rrf_eq_k10_top20`
  - `doc@4 = 2346 / 2441`
  - `doc@20 = 2401 / 2441`
- `rrf_sparseheavy_k10_top20`
  - `doc@4 = 2350 / 2441`
  - `doc@20 = 2400 / 2441`
- `rrf_eq_k5_top20`
  - `doc@4 = 2347 / 2441`
  - `doc@20 = 2401 / 2441`

Main interpretation:

- `splade_top20` already beats dense at both `doc@4` and `doc@20`
- the static `dense7+sparse8` hybrid is only a shortlist/coverage improver
  - it helps `doc@20`
  - it does not improve `doc@4`
- doc-level RRF is the first non-heuristic method here that improves both:
  - early ranking
  - shortlist coverage

### Best current non-heuristic retrieval settings

If early ranking is the main objective:

- use `rrf_sparseheavy_k10_top20`

Parameters:

- `--fusion-mode doc_rrf`
- `--dense-top-docs 20`
- `--sparse-top-docs 20`
- `--final-top-docs 20`
- `--rrf-k 10`
- `--dense-weight 0.75`
- `--sparse-weight 1.25`

Why:

- strongest overall `doc@4`
- still only `1` qid behind the best `doc@20` setting

If shortlist coverage is the only objective:

- use either:
  - `rrf_eq_k10_top20`
  - `rrf_eq_k5_top20`

because both reached:

- `2401 / 2441` at `doc@20`

But the current best single default is:

- `rrf_sparseheavy_k10_top20`

### Image-related qtype findings on full MMQA dev

#### Static hybrid at `doc@20`

The static hybrid was still useful as a recall improver on image-related qtypes:

- `ImageListQ`
  - `0.7163 -> 0.7730`
- `Compose(TextQ,ImageListQ)`
  - `0.7826 -> 0.8261`
- `Compose(ImageQ,TableQ)`
  - `0.9577 -> 0.9930`
- `Compare(Compose(TableQ,ImageQ),TableQ)`
  - `0.9904 -> 1.0000`

But at `doc@4`, the same static hybrid was effectively just the dense head:

- `hybrid_minus_dense = 0` on the checked qtypes

So it should be treated as:

- a shortlist recall improver
- not an early-ranking improver

#### RRF at `doc@4`

The RRF runs improved image-related early ranking:

- `ImageListQ`
  - dense `81 / 141`
  - SPLADE `76 / 141`
  - best RRF `85 / 141`
    - `rrf_eq_k10_top20`
- `ImageQ`
  - dense and SPLADE `227 / 230`
  - best RRF `229 / 230`
- `Compose(TextQ,ImageListQ)`
  - dense `29 / 46`
  - SPLADE `31 / 46`
  - best RRF `36 / 46`
    - `rrf_eq_k10_top20`
- `Compose(TableQ,ImageListQ)`
  - dense `179 / 195`
  - SPLADE `192 / 195`
  - best RRF `193 / 195`
    - `rrf_sparseheavy_k10_top20`
- `Compose(ImageQ,TableQ)`
  - dense `127 / 142`
  - SPLADE `139 / 142`
  - best RRF `141 / 142`
    - `rrf_sparseheavy_k10_top20`
- `Compose(ImageQ,TextQ)`
  - RRF tied dense at `19 / 20`
- `Compare(Compose(TableQ,ImageQ),Compose(TableQ,TextQ))`
  - dense `9 / 15`
  - SPLADE `14 / 15`
  - best RRF tied SPLADE at `14 / 15`

Interpretation:

- the earlier statement that SPLADE is weak at `ImageListQ` top-4 remains true
- but RRF fixes that weakness while keeping or improving SPLADE's gains elsewhere

### Important evaluation caveat

For full MMQA dev:

- automatic page-level evaluation is not meaningful
- the dataset does not provide reliable gold page labels for the whole split

So for full-dev comparisons:

- trust `doc@4`
- trust `doc@20`
- ignore the `page@k` fields unless the qids were manually reviewed and annotated

### Exact reproduction commands used for the best full-dev RRF run

Dense baseline prediction:

- `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json`

SPLADE prediction:

- `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_splade_mmqa_dev/mmqa_dev_splade.prediction.json`

Best default RRF command:

```bash
REPO=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
OUTDIR=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_splade_mmqa_dev

cd "${REPO}"
python scripts/fuse_page_retrieval_predictions.py \
  --dense-prediction-json /mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json \
  --sparse-prediction-json "${OUTDIR}/mmqa_dev_splade.prediction.json" \
  --gold "${REPO}/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl" \
  --fusion-mode doc_rrf \
  --dense-top-docs 20 \
  --sparse-top-docs 20 \
  --final-top-docs 20 \
  --rrf-k 10 \
  --dense-weight 0.75 \
  --sparse-weight 1.25 \
  --output-prediction-json "${OUTDIR}/mmqa_dev_rrf_sparseheavy_k10_top20.prediction.json" \
  --output-summary-json "${OUTDIR}/mmqa_dev_rrf_sparseheavy_k10_top20.summary.json"
```

### Recommended next step on other datasets

This is the direction to continue in a fresh chat.

Recommended order:

1. pick a target dataset
2. check whether its page corpus is already covered by an existing SPLADE page-text export and index
3. if not, rerun:
   - `scripts/export_m3docvqa_page_text.py`
   - `scripts/build_splade_page_index.py`
4. generate:
   - dense retrieval prediction
   - SPLADE retrieval prediction
5. fuse with:
   - `--fusion-mode doc_rrf`
   - `--rrf-k 10`
   - `--dense-weight 0.75`
   - `--sparse-weight 1.25`
6. evaluate:
   - overall `doc@4`
   - overall `doc@20`
   - qtype or subset breakdown if the dataset has mixed question types

Questions to answer on the next dataset:

- does RRF still beat both dense and SPLADE at `doc@4`?
- does it keep a `doc@20` gain too?
- is there an `ImageListQ`-like subtype where dense alone is still special?
- does the sparse-heavy weighting remain best, or should the balance be retuned?

### Current recommended narrative

The strongest current narrative is no longer:

- visual-aware reranking beats the main retriever

The strongest current narrative is:

- a non-heuristic dense+sparse document-level fusion improves retrieval quality
- static dense+sparse tail-union helps recall but not early ranking
- doc-level RRF improves both early ranking and shortlist coverage
- heuristic routing is interesting on `ImageListQ`, but RRF is the cleaner portable method to carry to other datasets

## 2026-05-21 Graph PPR transfer handoff

### Why this section exists

The next goal is to verify whether the graph-PPR dense+sparse reranker generalizes beyond M3DocVQA/MMQA dev.

This is the method to carry into another chat or another dataset. The important point is that the method is not a visual-reranker-specific trick:

- it needs one dense page retriever prediction file
- it needs one sparse/SPLADE page retriever prediction file
- it builds a small query-local graph over candidate pages and their owning docs
- it runs PPR on that graph
- it emits a doc-shortlist-style prediction by keeping one page per doc

The current best setting improved full MMQA dev from:

- dense `doc@4 = 2277 / 2441`, `doc@20 = 2368 / 2441`
- SPLADE `doc@4 = 2329 / 2441`, `doc@20 = 2391 / 2441`
- page/doc RRF baseline `doc@4 = 2348 / 2441`, `doc@20 = 2402 / 2441`

to:

- graph PPR `doc@4 = 2363 / 2441`, `doc@20 = 2406 / 2441`

### Code and branch

Use branch:

- `codex/mmdocir-hpc-workflow`

Key scripts:

- `scripts/graph_rerank_page_retrieval_predictions.py`
- `scripts/run_m3docvqa_graph_ppr_experiments.sh`
- `scripts/summarize_graph_ppr_summaries.py`

Most relevant commits:

- `9000dd0` Add graph PPR retrieval reranker
- `22c91af` Add SPLADE expansion path for graph PPR reranking
- `97aba7c` Add graph PPR source ablations
- `d6e2bb8` Add graph PPR ablation sweeps
- `9554e60` Add graph PPR budget comparison ablations
- `d0242ec` Add graph PPR final reporting helpers

### Required input schema for another dataset

The graph script expects two prediction JSONs keyed by qid:

```json
{
  "qid-1": {
    "question": "...",
    "page_retrieval_results": [
      ["doc_id_a", 0, 12.34],
      ["doc_id_b", 2, 11.98]
    ]
  }
}
```

Requirements:

- dense and sparse prediction files must share qids
- page rows must be `[doc_id, page_idx, score]`
- `page_idx` should be zero-based and stable across dense, sparse, and gold
- doc IDs must match gold `supporting_context[].doc_id`
- scores can be arbitrary because the default seed is rank/RRF-based
- for doc-level evaluation, gold JSONL should contain `qid` and `supporting_context` entries with `doc_id`
- for qtype filtering, gold JSONL should contain `metadata.type`
- page-level metrics are optional and require either `metadata.gold_page_uids` or `supporting_context[].page_idx`

For datasets without reliable page labels, still run the method and trust:

- `reranked_top4_doc_count`
- `reranked_top20_doc_count`
- `candidate_gold_doc_count`
- `candidate_gold_doc_miss_count`

### Graph method details

For each qid:

1. take top dense pages and top sparse pages
2. union them into a candidate page pool
3. add one node per candidate page
4. add one node per candidate document
5. seed page nodes with dense/sparse RRF source scores
6. optionally seed doc nodes, but the best config uses no doc seed
7. connect each page to its owning doc
8. optionally connect adjacent candidate pages from the same doc
9. run PPR
10. score pages with:
    - normalized source page seed
    - normalized page PPR
    - owning-doc PPR
11. output top pages, usually with `--per-doc-page-limit 1` for doc ranking

Important finding:

- page-doc edges are the useful graph structure
- adjacent-page edges had almost no effect
- doc seed hurt or was unnecessary; best configs use `--doc-seed-weight 0.0`

### Naming: Graph PPR, graph1000, and page-RRF1000

Use these names precisely in future notes:

| name | meaning | graph? | PPR? | candidate pool |
| --- | --- | --- | --- | --- |
| `Graph PPR` | the general algorithm family | yes | yes | configurable dense/sparse pages |
| `graph1000` | best full Graph PPR config | yes | yes | dense top1000 + SPLADE top1000 |
| `graph100` | efficient Graph PPR config | yes | yes | dense top100 + SPLADE top100 |
| `page-RRF1000` | matched no-graph baseline | no | no | dense top1000 + SPLADE top1000 |
| `page-RRF100` | matched efficient no-graph baseline | no | no | dense top100 + SPLADE top100 |

The `1000` or `100` number is the input candidate budget per source, not the output size.

For all final doc-shortlist comparisons here:

- output is still `--final-top-pages 20`
- output uses `--per-doc-page-limit 1`
- so the result behaves like a top-20 document shortlist

The key experimental control is:

- `page-RRF1000` and `graph1000` use the same dense top1000 + SPLADE top1000 candidate pool
- `page-RRF1000` scores pages only by rank-level RRF seed
- `graph1000` adds page/doc graph propagation and final PPR-based scoring

So the difference between `page-RRF1000` and `graph1000` isolates the value of the graph step beyond simple dense+sparse page fusion.

The page-RRF seed is:

```text
seed(page) = dense_weight / (rrf_k + dense_rank)
           + sparse_weight / (rrf_k + sparse_rank)
```

The best Graph PPR final score is:

```text
final_score(page) =
  1.0  * normalized_page_seed
+ 1.5  * normalized_page_ppr
+ 0.75 * normalized_owning_doc_ppr
```

Intuition:

- page-RRF treats pages mostly independently
- Graph PPR lets evidence move from page nodes to doc nodes and back to pages
- this lets a document accumulate support from several moderately ranked pages
- the owning-doc PPR term can lift the best page from that document into the final one-page-per-doc shortlist

### Best default config for transfer

Use this as the main result config on a new dataset:

```bash
python scripts/graph_rerank_page_retrieval_predictions.py \
  --dense-prediction-json "${DENSE_PRED}" \
  --sparse-prediction-json "${SPLADE_PRED}" \
  --gold "${GOLD_JSONL}" \
  --question-type "${QUESTION_TYPE}" \
  --dense-top-pages 1000 \
  --sparse-top-pages 1000 \
  --final-top-pages 20 \
  --per-doc-page-limit 1 \
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
  --output-prediction-json "${OUTDIR}/graph_ppr_best_top20.prediction.json" \
  --output-summary-json "${OUTDIR}/graph_ppr_best_top20.summary.json"
```

Best config name used in the M3DocVQA/MMQA experiments:

- `fulldev_graph_ppr_budget1000_top20_nodocseed_restart0p15_pagew1p5_docw0p75`
- equivalent final-report label:
  - `fulldev_graph_ppr_final_graph1000_top20_nodocseed_restart0p15_pagew1p5_docw0p75`

Best full-dev result:

- `doc@4 = 2363 / 2441`
- `doc@20 = 2406 / 2441`
- candidate gold docs: `2439 / 2441`
- candidate misses: `2`
- mean candidate pages: `1774.862`
- mean candidate docs: `814.881`

Matched paired comparison against `page-RRF1000`:

| comparison | doc@4 | net doc@4 | candidate-only | baseline-only | sign-test p | doc@20 | net doc@20 | doc@20 p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `graph1000` vs `page-RRF1000` | `2363` vs `2348` | `+15` | `18` | `3` | `0.00149` | `2406` vs `2402` | `+4` | `0.125` |

Interpretation:

- full Graph PPR significantly improves early document ranking over the matched page-RRF baseline
- the `doc@20` effect is positive but not significant under the same paired sign test
- this is the strongest evidence that the graph step matters beyond rank-only dense+sparse fusion

### Efficient config to try first when runtime matters

Use this when a dataset is large and the full 1000/1000 candidate graph is too expensive:

```bash
python scripts/graph_rerank_page_retrieval_predictions.py \
  --dense-prediction-json "${DENSE_PRED}" \
  --sparse-prediction-json "${SPLADE_PRED}" \
  --gold "${GOLD_JSONL}" \
  --question-type "${QUESTION_TYPE}" \
  --dense-top-pages 100 \
  --sparse-top-pages 100 \
  --final-top-pages 20 \
  --per-doc-page-limit 1 \
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
  --output-prediction-json "${OUTDIR}/graph_ppr_budget100_top20.prediction.json" \
  --output-summary-json "${OUTDIR}/graph_ppr_budget100_top20.summary.json"
```

Full-dev efficient result:

- `doc@4 = 2354 / 2441`
- `doc@20 = 2404 / 2441`
- candidate gold docs: `2428 / 2441`
- candidate misses: `13`
- mean candidate pages: `174.605`
- mean candidate docs: `103.816`

Interpretation:

- this is only `-9 doc@4` and `-2 doc@20` behind the full 1000/1000 graph
- it uses roughly one tenth of the candidate pages
- it still beats full-budget page RRF at `doc@4`

### Best asymmetric budget configs

If only one branch can be large, keep SPLADE large and reduce dense first:

| config | doc@4 | doc@20 | mean pages | mean docs |
| --- | ---: | ---: | ---: | ---: |
| dense20/sparse1000 | 2352 | 2402 | 1005.133 | 457.731 |
| dense50/sparse1000 | 2358 | 2404 | 1018.959 | 463.796 |
| dense100/sparse1000 | 2359 | 2405 | 1047.010 | 475.991 |
| dense200/sparse1000 | 2360 | 2405 | 1111.655 | 503.881 |
| dense500/sparse1000 | 2359 | 2406 | 1337.298 | 601.735 |
| dense1000/sparse1000 | 2363 | 2406 | 1774.862 | 814.881 |

The reverse direction was also useful but a little weaker at top-4:

| config | doc@4 | doc@20 | mean pages | mean docs |
| --- | ---: | ---: | ---: | ---: |
| dense1000/sparse20 | 2353 | 2401 | 1004.853 | 580.223 |
| dense1000/sparse50 | 2355 | 2403 | 1018.753 | 584.937 |
| dense1000/sparse100 | 2355 | 2405 | 1048.066 | 594.683 |
| dense1000/sparse200 | 2358 | 2405 | 1116.354 | 617.388 |
| dense1000/sparse500 | 2360 | 2406 | 1349.511 | 691.604 |
| dense1000/sparse1000 | 2363 | 2406 | 1774.862 | 814.881 |

Recommendation:

- for best accuracy: `1000/1000`
- for efficient strong accuracy: `100/100`
- for a middle ground: `dense100/sparse1000` or `dense200/sparse1000`

### Baselines and controls to rerun on a new dataset

Run these on every new dataset:

1. dense only
2. SPLADE only
3. page/doc RRF baseline, no graph propagation
4. graph PPR efficient `100/100`
5. graph PPR full `1000/1000`
6. graph PPR with no page-doc edges
7. seed-only / page-RRF component control
8. page-PPR-only and doc-PPR-only component controls if time permits

The RRF baseline command uses the same graph script but disables graph propagation:

```bash
python scripts/graph_rerank_page_retrieval_predictions.py \
  --dense-prediction-json "${DENSE_PRED}" \
  --sparse-prediction-json "${SPLADE_PRED}" \
  --gold "${GOLD_JSONL}" \
  --question-type "${QUESTION_TYPE}" \
  --dense-top-pages 1000 \
  --sparse-top-pages 1000 \
  --final-top-pages 20 \
  --per-doc-page-limit 1 \
  --doc-seed-weight 0.0 \
  --ppr-iters 0 \
  --page-doc-edge-weight 0.0 \
  --same-doc-window 0 \
  --adjacent-page-edge-weight 0.0 \
  --final-page-seed-weight 1.0 \
  --final-ppr-page-weight 0.0 \
  --final-ppr-doc-weight 0.0 \
  --output-prediction-json "${OUTDIR}/page_rrf1000_top20.prediction.json" \
  --output-summary-json "${OUTDIR}/page_rrf1000_top20.summary.json"
```

### Source ablation results on full MMQA dev

These show why the final method should use the plain top224 dense branch plus SPLADE.

| source setup | dense source doc@4 | sparse source doc@4 | reranked doc@4 | reranked doc@20 | mean pages | mean docs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| plain top224 dense only, no SPLADE | 2277 | 0 | 2280 | 2368 | 1000.000 | 578.636 |
| raw dense only | 2193 | 0 | 2213 | 2345 | 1000.000 | 578.636 |
| raw dense + SPLADE | 2193 | 2329 | 2348 | 2401 | 1774.862 | 814.881 |
| SPLADE only | 0 | 2329 | 2334 | 2391 | 1000.000 | 455.549 |
| plain top224 dense + SPLADE, best graph | 2277 | 2329 | 2363 | 2406 | 1774.862 | 814.881 |

Interpretation:

- SPLADE alone is strong
- raw dense is weaker than the plain top224 dense branch
- plain top224 dense + SPLADE is the best source combination
- graph propagation adds value beyond simply replacing dense with SPLADE

### Symmetric budget ablation results

| dense pages | sparse pages | graph doc@4 | graph doc@20 | page-RRF doc@4 | page-RRF doc@20 | candidate gold docs | mean pages | mean docs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 20 | 2355 | 2399 | 2349 | 2399 | 2403 | 32.916 | 18.801 |
| 50 | 50 | 2352 | 2401 | 2348 | 2402 | 2414 | 85.524 | 51.401 |
| 100 | 100 | 2354 | 2404 | 2350 | 2404 | 2428 | 174.605 | 103.816 |
| 200 | 200 | 2353 | 2404 | 2349 | 2402 | 2429 | 353.315 | 200.142 |
| 500 | 500 | 2359 | 2405 | 2348 | 2402 | 2436 | 884.465 | 446.957 |
| 1000 | 1000 | 2363 | 2406 | 2348 | 2402 | 2439 | 1774.862 | 814.881 |

Interpretation:

- graph PPR is consistently better than matched-budget page RRF at `doc@4`
- the full budget is best overall
- the 100/100 budget is the best efficiency point
- candidate recall saturates quickly but top-4 promotion still benefits from more candidates

### Graph structure ablations

| structure | doc@4 | doc@20 | recover vs dense@4 | lose vs dense@4 |
| --- | ---: | ---: | ---: | ---: |
| full graph | 2363 | 2406 | 93 | 7 |
| no adjacent-page edges | 2362 | 2406 | 93 | 8 |
| no page-doc edges | 2343 | 2404 | 84 | 18 |
| no edges / seed only | 2348 | 2402 | 84 | 13 |

Interpretation:

- page-doc edges are the important graph signal
- adjacent-page edges are not important for this dataset
- removing all edges collapses to the page-RRF baseline

For transfer:

- keep `--page-doc-edge-weight 1.0`
- keep adjacent edges for the default run, but do not over-claim them
- rerun `no_page_doc` as the most important graph-structure ablation

### Score component ablations

| scoring mode | doc@4 | doc@20 | recover vs dense@4 | lose vs dense@4 |
| --- | ---: | ---: | ---: | ---: |
| full: seed + page PPR + doc PPR | 2363 | 2406 | 93 | 7 |
| PPR without final seed | 2361 | 2406 | 93 | 9 |
| page-PPR only | 2354 | 2402 | 88 | 11 |
| seed only / page RRF | 2348 | 2402 | 84 | 13 |
| doc-PPR only | 2328 | 2407 | 91 | 40 |

Interpretation:

- full scoring is best for `doc@4`
- final source seed is helpful but not the main driver
- page PPR alone helps over seed-only
- doc PPR alone can improve `doc@20` but hurts early ranking badly
- the best score is the combination, not a single component

### Final reporting commands on M3DocVQA/MMQA dev

Use these to regenerate clean final summaries with consistent oracle fields:

```bash
git pull --rebase --autostash origin codex/mmdocir-hpc-workflow

QUESTION_TYPE= LABEL_PREFIX=fulldev \
RUN_DOC_TOP20=0 RUN_PAGE_TOP500=0 RUN_FINAL_REPORT_CONFIGS=1 \
BEST_RESTART_PROB=0.15 BEST_PAGE_PPR_WEIGHT=1.5 BEST_DOC_PPR_WEIGHT=0.75 \
scripts/run_m3docvqa_graph_ppr_experiments.sh

QUESTION_TYPE= LABEL_PREFIX=fulldev \
RUN_DOC_TOP20=0 RUN_PAGE_TOP500=0 RUN_FINAL_QTYPE_SWEEP=1 \
BEST_RESTART_PROB=0.15 BEST_PAGE_PPR_WEIGHT=1.5 BEST_DOC_PPR_WEIGHT=0.75 \
scripts/run_m3docvqa_graph_ppr_experiments.sh
```

Summarize:

```bash
OUTDIR=/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_splade_mmqa_dev

python scripts/summarize_graph_ppr_summaries.py --format markdown \
  "$OUTDIR"/fulldev_graph_ppr_final_*.summary.json

python scripts/summarize_graph_ppr_summaries.py --format markdown \
  "$OUTDIR"/fulldev_graph_ppr_qtype_*.summary.json
```

Paired comparison against page RRF:

```bash
python scripts/summarize_graph_ppr_summaries.py \
  --compare "$OUTDIR"/fulldev_graph_ppr_final_page_rrf1000_top20.summary.json \
            "$OUTDIR"/fulldev_graph_ppr_final_graph1000_top20_nodocseed_restart0p15_pagew1p5_docw0p75.summary.json
```

Efficient graph comparison:

```bash
python scripts/summarize_graph_ppr_summaries.py \
  --compare "$OUTDIR"/fulldev_graph_ppr_final_page_rrf1000_top20.summary.json \
            "$OUTDIR"/fulldev_graph_ppr_final_graph100_top20_nodocseed_restart0p15_pagew1p5_docw0p75.summary.json
```

### Minimal transfer experiment plan for another dataset

Run this exact sequence in the next dataset:

1. dense baseline
   - record `doc@4`, `doc@20`
2. SPLADE baseline
   - record `doc@4`, `doc@20`
3. page-RRF baseline at `1000/1000`
   - confirms dense+sparse fusion without graph propagation
4. graph PPR at `100/100`
   - efficient transfer test
5. graph PPR at `1000/1000`
   - best-accuracy transfer test
6. no-page-doc graph ablation
   - tests whether document nodes matter on the new dataset
7. seed-only / page-RRF component control
   - should match or nearly match the no-edge result

Only run the larger sweeps if the main transfer test is positive:

- budget sweep: `20 50 100 200 500 1000`
- asymmetric budgets: dense-small/sparse1000 and dense1000/sparse-small
- component sweep: full, seed-only, page-PPR-only, doc-PPR-only, PPR-no-final-seed
- restart sweep: `0.05 0.10 0.15 0.20 0.25 0.35`

### What to verify on other datasets

Main questions:

- does graph PPR beat dense alone at `doc@4`?
- does graph PPR beat SPLADE alone at `doc@4`?
- does graph PPR beat page/doc RRF at `doc@4`?
- does the gain persist at `doc@20`?
- is `100/100` still close to `1000/1000`?
- are page-doc edges still the critical graph structure?
- is the best restart still near `0.15`?
- does doc-PPR-only still hurt early ranking?

Failure modes to watch:

- candidate recall is too low, shown by low `candidate_gold_doc_count`
- sparse branch dominates and dense adds little
- dense branch dominates and sparse adds little
- graph improves `doc@20` but hurts `doc@4`
- page-level gold is unreliable, causing confusing page metrics

### Current claim to make cautiously

The safe claim after M3DocVQA/MMQA dev is:

- dense + SPLADE page candidates are complementary
- rank-only page RRF is a strong baseline
- adding query-local page-doc graph propagation significantly improves early document ranking over matched page-RRF at full `1000/1000` budget
- the improvement comes mainly from page-doc propagation and the combined seed/page-PPR/doc-PPR score
- the method is portable enough to test on other document VQA datasets because it only needs dense and sparse page retrieval outputs

Do not yet claim:

- the method is universally better than all dense+sparse fusion methods
- adjacent-page edges are important
- page-level metrics are meaningful on full MMQA dev
- the same weights are guaranteed optimal on another dataset

### External runner profile check

The reusable external-dataset wrapper is:

- `scripts/run_external_graph_ppr_pipeline.sh`

It now has two explicit profiles:

1. `GRAPH_PROFILE=doc_shortlist_best`
   - this is the default
   - matches the best M3DocVQA/MMQA transfer config above
   - uses:
     - `DENSE_TOP_PAGES=1000`
     - `SPARSE_TOP_PAGES=1000`
     - `FINAL_TOP_PAGES=20`
     - `PER_DOC_PAGE_LIMIT=1`
     - `DOC_SEED_WEIGHT=0.0`
     - `RESTART_PROB=0.15`
     - `FINAL_PAGE_SEED_WEIGHT=1.0`
     - `FINAL_PPR_PAGE_WEIGHT=1.5`
     - `FINAL_PPR_DOC_WEIGHT=0.75`
   - use this for the main transfer claim
2. `GRAPH_PROFILE=page_rank_probe`
   - uses:
     - `FINAL_TOP_PAGES=1000`
     - `PER_DOC_PAGE_LIMIT=0`
     - `RESTART_PROB=0.20`
     - `FINAL_PPR_DOC_WEIGHT=0.5`
   - use this only to inspect page ranking behavior when reliable page labels exist

This distinction matters. The first SciEGQA/ViDoSeek external runs used the page-ranking probe profile, so they are encouraging sanity checks but not the exact handoff-best document-shortlist transfer config.

Observed page-ranking-probe results:

| dataset | qids | page@4 | page@20 | doc@4 | doc@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SciEGQA-Bench | 1623 | 0.7686 | 0.9091 | 0.9298 | 0.9852 |
| ViDoSeek | 1142 | 0.8905 | 0.9982 | 0.9991 | 1.0000 |

The same external datasets were then rerun with `GRAPH_PROFILE=doc_shortlist_best`. The saved summaries were checked and confirmed to use the exact intended config:

```text
dense_top_pages = 1000
sparse_top_pages = 1000
final_top_pages = 20
per_doc_page_limit = 1
rrf_k = 10
dense_weight = 1.0
sparse_weight = 1.0
doc_seed_weight = 0.0
restart_prob = 0.15
ppr_iters = 30
page_doc_edge_weight = 1.0
same_doc_window = 1
adjacent_page_edge_weight = 0.25
final_page_seed_weight = 1.0
final_ppr_page_weight = 1.5
final_ppr_doc_weight = 0.75
```

Observed `doc_shortlist_best` transfer results:

| dataset | qids | doc@4 | doc@20 | page@4 | page@20 |
| --- | ---:| ---: | ---: | ---: | ---: |
| MMDocIR | 1658 | 0.8034 | 0.8884 | 0.4562 | 0.4998 |
| SciEGQA-Bench | 1623 | 0.9279 | 0.9846 | 0.5173 | 0.5474 |
| ViDoSeek | 1142 | 0.9991 | 1.0000 | 0.6743 | 0.6751 |
| ViDoRe V3 | 14514 | 0.8725 | 0.9703 | 0.2064 | 0.2285 |

External-transfer conclusion:

- The M3DocVQA best Graph-PPR document-shortlist config transferred correctly at the implementation/config level.
- It did not transfer as the best retrieval method on the external datasets.
- The likely reason is objective mismatch: `doc_shortlist_best` emits one representative page per document, while MMDocIR, SciEGQA, ViDoSeek, and ViDoRe V3 evaluate exact page retrieval heavily.
- On MMDocIR and ViDoRe V3, the method trails `plain_top224` at early document recall and is much worse at page recall.
- On SciEGQA-Bench, it improves document recall over `plain_top224`, but the page-ranking probe profile is far better for page recall.
- On ViDoSeek, document recall is already saturated, so the small document gain is not useful enough to justify the page-recall loss.

Current recommendation after transfer:

- Use `doc_shortlist_best` only when the downstream stage needs a document shortlist or one representative page per document.
- For external page-labeled datasets, use `GRAPH_PROFILE=page_rank_probe` or tune a page-preserving Graph-PPR variant.
- The next useful external ablation is not another doc-shortlist run; it is a page-preserving sweep over:
  - `RESTART_PROB`
  - `FINAL_PPR_DOC_WEIGHT`
  - `FINAL_PPR_PAGE_WEIGHT`
  - optional `PER_DOC_PAGE_LIMIT=0` vs small per-doc caps greater than 1.

Follow-up page-preserving results changed the external-dataset conclusion. The best frozen general page-labeled config observed so far is `denseheavy_lightboth`:

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

Observed `denseheavy_lightboth` versus `plain_top224`:

| dataset | qids | page@1 | page@4 | page@20 | doc@4 | doc@20 | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SciEGQA-Bench | 1623 | 0.4951 *(plain 0.5228)* | 0.7686 *(plain 0.7394)* | 0.9104 *(plain 0.8758)* | 0.9261 *(plain 0.9070)* | 0.9852 *(plain 0.9772)* | wins at page@4/@20; loses page@1 |
| MMDocIR | 1658 | 0.4074 *(plain 0.4136)* | 0.6342 *(plain 0.6075)* | 0.7662 *(plain 0.7480)* | 0.8148 *(plain 0.8058)* | 0.8920 *(plain 0.8890)* | wins at page@4/@20; loses page@1 |
| ViDoRe V3 | 14514 | 0.1689 *(plain 0.1730)* | 0.3475 *(plain 0.3312)* | 0.5706 *(plain 0.5431)* | 0.8959 *(plain 0.8854)* | 0.9751 *(plain 0.9809)* | wins page@4/@20; loses page@1/doc@20 |
| ViDoSeek | 1142 | 0.6567 *(plain 0.6830)* | 0.8923 *(plain 0.8958)* | 0.9974 *(plain 0.9842)* | 1.0000 *(plain 0.9982)* | 1.0000 *(plain 1.0000)* | saturated; plain still slightly better at page@1/@4 |

Updated recommendation:

- For exact page-labeled datasets, the strongest general transfer is page-preserving dense-heavy light Graph-PPR, not the M3DocVQA document-shortlist profile.
- The claim should be framed around practical context depths: page@4/page@10/page@20. Do not claim a universal rank-1 win.
- ViDoSeek is the main exception for page@4: `plain_top224` is still slightly stronger there, while dense-heavy light Graph-PPR improves deeper recall.
- Keep `doc_shortlist_best` for M3DocVQA/MMQA-style document shortlist retrieval, where exact gold page labels are not the main target.
