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

Best fixed weights found on the cleanest recent useful run:

- `base = 1.0`
- `visual = 2.0`
- `non_visual = 1.0`
- `balance = 2.0`

This is the recommended first fixed-weight transfer setting for the next qid before doing qid-specific grid search.

## Recommended next experiment

Use a fixed-weight transfer run on:

- `3444052221c1104c977a4653988d44f1`

Suggested command:

```bash
python scripts/rerank_target_docs_visual_aware.py \
  --qid 3444052221c1104c977a4653988d44f1 \
  --embedding_name colpali-v1.2_m3-docvqa_dev \
  --query_token_filter drop_pad_like \
  --splice-query-token-labels /mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/visual_needed_binary/deberta_v3_large_seed42/export/dev_query_visual_binary_labels_union_relaxed_v2.jsonl \
  --splice-patch-labels-jsonl /mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/layout_patch_assignments_done_so_far_plus_new_3class_full.jsonl \
  --baseline-pred /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output/retrieval_only_dev_ret1000full_drop_pad_like/colpali-v1.2_ivfflat_ret1000_qtf-drop_pad_like_2026-04-16_07-21-44.json \
  --from-baseline-top-pages 1000 \
  --weight-base 1.0 \
  --weight-visual 2.0 \
  --weight-non-visual 1.0 \
  --weight-balance 2.0 \
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

