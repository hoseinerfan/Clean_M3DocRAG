# Page Evidence Promotion Findings 2026-06-02

Updated 2026-06-03 with the clean out-of-fold hybrid result.

Purpose: consolidate the current M3DocVQA/MMQA pseudo-page supervision direction, the strongest page-promotion results, the zero-shot transfer results on page-labeled external datasets, and the current interpretation for thesis/advisor reporting.

## Current Thesis Direction

The current strongest direction is a graph-aware page evidence promotion framework for multimodal document QA. The framework:

1. derives pseudo-page supervision from MMQA evidence and exported page text,
2. diagnoses whether failures are caused by missing evidence pages from the top-1000 pool or by failing to promote them into top-k,
3. trains and evaluates page-level promotion methods that use dense, sparse, graph-propagation, hyperlink, and content evidence signals.

The important reframing is:

> Given a large retrieved candidate pool, learn how to promote the most useful evidence pages into the small top-k context used by downstream QA.

This is no longer only a graph-link discovery problem. Many evidence pages are already in the candidate pool; the harder problem is safe promotion.

## Pseudo-Page Labels

The original M3DocVQA/MMQA files mostly support document-level evaluation. We built pseudo-page labels from MMQA evidence, page text, tables, images, and document mappings so page-level retrieval can be evaluated and used for training.

Strict pseudo-label setting:

| Split | qids | matched qids | matched fraction | pseudo-page labels |
|---|---:|---:|---:|---:|
| dev | 2,441 | 2,285 | 0.9361 | 3,649 |
| train | 23,817 | 22,389 | 0.9400 | 34,637 |

These labels are required for M3DocVQA training/tuning, but not for inference on other datasets. For zero-shot transfer, the M3DocVQA-trained model only needs each target dataset's candidate pages, page text, and retrieval/graph source predictions.

## Main M3DocVQA/MMQA Result

Evaluation: M3DocVQA/MMQA dev with strict pseudo-page labels. The final proposed method is **OOF content-aware page evidence promotion + counterfactual top-4 boundary repair**.

Advisor-facing strict pseudo-page dev table:

| Method | page@4 | page@5 | Gain@5 vs dense | page@10 | doc@4 | Role |
|---|---:|---:|---:|---:|---:|---|
| Dense baseline | 0.6210 | 0.6556 | +0.0000 | 0.7383 | 0.9160 | Dense page pool baseline. |
| GPP no-hyperlink | 0.6740 | 0.7221 | +0.0665 | 0.8328 | 0.9488 | Graph baseline without hyperlink propagation. |
| GPP doc-hyperlink | 0.6687 | 0.7138 | +0.0582 | 0.8280 | 0.9514 | Hyperlink variant; better document recall than page recall. |
| GPP page-hyperlink | 0.6709 | 0.7177 | +0.0621 | 0.8298 | 0.9510 | Hyperlink page-node variant. |
| Content-aware promotion, adaptive `page@5` | 0.7659 | 0.8031 | +0.1475 | 0.8687 | 0.9545 | Strong first-stage page evidence promotion. |
| Practical hybrid insert-rank-4 | 0.7729 | 0.8144 | +0.1588 | 0.8748 | 0.9558 | Practical hybrid trained using full-train content-aware base. |
| **OOF hybrid insert-rank-4** | **0.7764** | **0.8179** | **+0.1623** | **0.8753** | **0.9562** | **Final thesis-grade method.** |

Interpretation:

- The final OOF hybrid improves `page@5` by `+0.1623` over dense retrieval, `+0.0958` over GPP no-hyperlink, and `+0.0149` over content-aware promotion alone.
- The OOF hybrid improves `page@4` by `+0.1554` over dense retrieval and `+0.0105` over content-aware promotion alone.
- The gain is page-evidence focused. It is not simply a document-shortlist improvement.
- Hyperlink GPP helps document-level recall, but page-level evidence selection still needs content-aware promotion.

## Adaptive Tuning

`TUNE_HIT_K=5` is a real held-out tuning step, not just a CLI label. The script:

1. splits train qids into fit/tune partitions,
2. trains the content-aware reranker on the fit partition,
3. sweeps blend strengths,
4. evaluates each blend on held-out train pseudo-page `page@5`,
5. selects the best blend,
6. retrains on all train qids and applies the selected blend to dev/test.

Observed adaptive settings:

| Tuning objective | selected alpha | behavior |
|---|---:|---|
| `page@4` | 0.45 | stronger early promotion |
| `page@5` | 0.40 | best top-5 evidence result |
| `page@10` | 0.30 | more conservative, better broader ranking |

Compared with the earlier fixed `alpha=0.30` content-aware result, adaptive `page@5` improved `page@4` and `page@5`, while giving up some `page@10` and doc-level recall:

| Method | page@4 | page@5 | page@10 | doc@4 |
|---|---:|---:|---:|---:|
| fixed `alpha=0.30` | 0.7593 | 0.7996 | 0.8827 | 0.9584 |
| adaptive `page@5`, `alpha=0.40` | 0.7659 | 0.8031 | 0.8687 | 0.9545 |

## Counterfactual Page Promotion

Counterfactual page promotion is a safety-oriented extension. Instead of reranking all candidates, it learns whether promoting one candidate page would repair the current top-k evidence set.

Top-5 safe-repair run:

- base: GPP no-hyperlink
- objective: repair `page@5`
- insertion rank: 5
- selected threshold: 0.80
- promoted pages on dev: 1,643
- movement: 188 recovered, 77 lost, net +111

| Method | page@4 | page@5 | page@10 | doc@4 | doc@5 |
|---|---:|---:|---:|---:|---:|
| GPP no-hyperlink base | 0.6740 | 0.7221 | 0.8328 | 0.9488 | 0.9540 |
| Counterfactual page promotion | 0.6740 | 0.7707 | 0.8530 | 0.9497 | 0.9584 |
| Gain | +0.0000 | +0.0486 | +0.0201 | +0.0009 | +0.0044 |

Top-4 repair run:

- base: GPP no-hyperlink
- objective: repair `page@4`
- insertion rank: 4
- selected threshold: 0.80
- promoted pages on dev: 1,702
- movement: 196 recovered, 72 lost, net +124

| Method | page@4 | page@5 | page@10 | doc@4 | doc@5 |
|---|---:|---:|---:|---:|---:|
| GPP no-hyperlink base | 0.6740 | 0.7221 | 0.8328 | 0.9488 | 0.9540 |
| Counterfactual page promotion, insert rank 4 | 0.7282 | 0.7694 | 0.8556 | 0.9510 | 0.9580 |
| Gain | +0.0543 | +0.0473 | +0.0228 | +0.0022 | +0.0039 |

Interpretation:

- The rank-5 variant is conservative: it improves `page@5` while preserving the original top-4 list.
- The rank-4 variant is more useful for evidence retrieval: it directly improves `page@4` by `+0.0543` and still improves `page@5` and `page@10`.
- Counterfactual promotion is more than a safe extension; it is a strong targeted repair method. When applied after content-aware promotion with clean OOF train predictions, it becomes the final best method.

## OOF Content-Aware + Counterfactual Hybrid

The practical hybrid was useful but not fully thesis-grade, because the train-side content-aware base was produced by a model trained on all train qids. The clean version uses out-of-fold content-aware train predictions:

1. split the 23,817 train qids into 5 deterministic folds,
2. train content-aware promotion on 4 folds,
3. predict the held-out fold,
4. merge the held-out predictions into one OOF train prediction covering all 23,817 train qids,
5. train counterfactual repair on this OOF content-aware train base,
6. evaluate on dev using the full-train content-aware dev prediction as the stage-1 base.

OOF sanity:

| Check | Value |
|---|---:|
| train qids covered by merged OOF prediction | 23,817 |
| fold held-out qids | 4,764 / 4,764 / 4,763 / 4,763 / 4,763 |
| selected threshold, insert-rank-4 | 0.80 |
| dev promoted actions, insert-rank-4 | 1,361 |
| movement vs content-aware, insert-rank-4 | 98 recovered, 74 lost |

Final OOF hybrid comparison:

| Method | page@4 | page@5 | page@10 | doc@4 | doc@5 |
|---|---:|---:|---:|---:|---:|
| Content-aware base | 0.7659 | 0.8031 | 0.8687 | 0.9545 | 0.9580 |
| OOF hybrid insert-rank-4 | 0.7764 | 0.8179 | 0.8753 | 0.9562 | 0.9593 |
| Gain | +0.0105 | +0.0149 | +0.0066 | +0.0018 | +0.0013 |
| OOF hybrid insert-rank-5 | 0.7659 | 0.8118 | 0.8748 | 0.9545 | 0.9593 |

Grouped result:

| Group | Method | page@4 | page@5 | page@10 | net@4 |
|---|---|---:|---:|---:|---:|
| single_gold_doc | Content-aware | 0.7969 | 0.8272 | 0.8827 | +93 |
| single_gold_doc | OOF hybrid insert-rank-4 | 0.8063 | 0.8356 | 0.8890 | +102 |
| multi_gold_doc | Content-aware | 0.7436 | 0.7857 | 0.8586 | +117 |
| multi_gold_doc | OOF hybrid insert-rank-4 | 0.7549 | 0.8053 | 0.8654 | +132 |

Interpretation:

- Insert-rank-4 is the final setting because it improves `page@4`, `page@5`, and `page@10`.
- Insert-rank-5 is a conservative ablation: it improves `page@5` while preserving `page@4`.
- The largest group gain is on multi-gold-doc questions, where evidence is more distributed and boundary repair is most useful.

## Graph-Aware LTR Result

True LightGBM LambdaMART was tested after installing `lightgbm==4.6.0`. It improved some early page promotion over its base but was too aggressive and did not beat content-aware promotion.

| Method | page@4 | page@5 | page@10 | doc@4 |
|---|---:|---:|---:|---:|
| GPP no-hyperlink base | 0.6740 | 0.7221 | 0.8328 | 0.9488 |
| LightGBM LambdaMART | 0.6954 | 0.7326 | 0.8026 | 0.9383 |

Interpretation: generic LTR is useful as a control but not sufficient as the main method. The stronger story is targeted evidence promotion, not standard ranking.

## Zero-Shot Transfer To Other Datasets

The M3DocVQA-trained adaptive content-aware model was applied to DUDE, MMDocIR, SciEGQA, and ViDoSeek without training on those datasets. This tests whether the learned evidence-promotion signal transfers.

Dense base transfer:

| Dataset | dense page@5 | transfer page@5 | gain | page@4 gain | page@10 gain | doc@4 gain |
|---|---:|---:|---:|---:|---:|---:|
| ViDoSeek | 0.9151 | 0.9229 | +0.0079 | +0.0070 | +0.0053 | +0.0000 |
| SciEGQA | 0.8065 | 0.8226 | +0.0160 | +0.0080 | +0.0117 | +0.0105 |
| DUDE | 0.5908 | 0.6173 | +0.0265 | +0.0269 | +0.0200 | +0.0107 |
| MMDocIR | 0.6580 | 0.6821 | +0.0241 | +0.0151 | +0.0115 | +0.0018 |

Interpretation:

- The M3DocVQA-trained model improves dense retrieval zero-shot on all four checked datasets.
- This supports the claim that the content/evidence promotion signal is not purely M3DocVQA-specific.
- The gains are strongest on DUDE and MMDocIR, and smallest on ViDoSeek because ViDoSeek is already saturated.

Docseed base transfer:

| Dataset | page@4 gain | page@5 gain | page@10 gain | doc@4 gain | Interpretation |
|---|---:|---:|---:|---:|---|
| ViDoSeek | +0.0061 | +0.0026 | +0.0053 | +0.0000 | small positive |
| SciEGQA | +0.0043 | +0.0111 | -0.0043 | +0.0018 | mixed |
| DUDE | -0.0086 | -0.0183 | -0.0541 | -0.0241 | negative |
| MMDocIR | -0.0320 | -0.0326 | -0.0513 | -0.0247 | negative |

Interpretation:

- The trained content-aware model is not a universal post-reranker after stronger docseed outputs.
- It works best as a dense-pool evidence promotion module.
- For DUDE and MMDocIR, `docseed_rrf_1p00` remains stronger than `docseed_rrf_1p00 + transfer`.

## Current Reporting Position

Advisor-safe claim:

> We propose a pseudo-page-supervised, graph-aware page evidence promotion framework. It constructs page-level supervision from MMQA evidence, shows that many failures are promotion failures inside the top-1000 candidate pool, and uses a two-stage model: content-aware evidence promotion followed by counterfactual top-4 boundary repair. The clean out-of-fold hybrid substantially improves page-level evidence recall on M3DocVQA/MMQA and the trained content-aware stage also transfers positively over dense retrieval on DUDE, MMDocIR, SciEGQA, and ViDoSeek.

Use these distinctions:

- **Main method:** OOF content-aware promotion + counterfactual insert-rank-4 boundary repair.
- **Stage-1 ablation:** content-aware promotion, adaptive `page@5`.
- **Repair ablations:** counterfactual promotion on GPP, practical hybrid, and conservative insert-rank-5.
- **Negative/secondary control:** standard LightGBM LambdaMART.
- **Transfer conclusion:** positive over dense bases; mixed/negative over stronger docseed bases.

## Next Steps

1. Run downstream answer-generation with the final OOF hybrid insert-rank-4 prediction and compare against dense, GPP, and content-aware-only.

2. Report trained transfer only as zero-shot over dense bases unless a safe router is added.

3. Consider a router that chooses between dense-pool content promotion and docseed output instead of stacking promotion after docseed.

4. Keep training-free self-calibrated promotion as a baseline idea, not the main novelty claim, unless it unexpectedly beats trained transfer across datasets.
