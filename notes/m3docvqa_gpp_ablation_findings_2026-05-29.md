# M3DocVQA GPP Ablation Findings 2026-05-29

Purpose: consolidate the M3DocVQA/MMQA Graph Page Preserve ablations run today, with separate tables for each experiment family, the current best config, and the artifact paths needed to reproduce or report the results.

## Metric Policy

- M3DocVQA/MMQA train and dev have `qid` and gold document labels, but no gold page labels.
- `doc@k` is the primary metric for this dataset.
- `row@k` is still useful because it measures the returned result rows after page selection and document diversity, but it is not exact page-gold recall.
- `net` in grouped tables is a movement diagnostic: `improved - worsened` by qid. It can be negative while `doc@k` or `row@k` is positive if the method helps more gold documents inside the top-k while moving the first gold hit slightly later for some qids.

## Current Best Config

Best current GPP hyperlink setting for M3DocVQA recall/context:

```bash
PDF_HYPERLINK_WEIGHT_MODE=log_count
PDF_HYPERLINK_TARGET_MODE=target_pages
PDF_HYPERLINK_TARGET_PAGES_PER_DOC=1
FINAL_SELECTION_MODE=mmr_doc_diverse
FINAL_SELECTION_NEW_DOC_BONUS=0.10
FINAL_SELECTION_SAME_DOC_PENALTY=0.05
```

Why this is the current default:

- It gives the best current returned-row context among the hyperlink-node runs: `row@4=0.804`, `row@20=0.905`.
- It gives the best deeper document recall among the same runs: `doc@10=0.911`, `doc@20=0.936`.
- It improves multi-gold-doc coverage strongly, which is the closest available proxy for cross-document evidence because M3DocVQA has no page-gold labels.
- Tradeoff: `doc@1`/`row@1` drop from `0.608` to `0.600`. If strict top-1 stability matters more, use `docnode_to_hyperlink_docs` instead.

## Important Paths

| Item | Path |
|---|---|
| M3DocVQA dev gold | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/data/m3-docvqa/multimodalqa/MMQA_dev.jsonl` |
| M3DocVQA train gold | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/data/m3-docvqa/multimodalqa/MMQA_train.jsonl` |
| Dense prediction input | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json` |
| SPLADE prediction input | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_splade_mmqa_dev/mmqa_dev_splade.prediction.json` |
| Page text JSONL | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_page_text/m3docvqa_dev_page_text.jsonl` |
| Mapped hyperlink edges | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/MMDocIR_M3DocRAG/output/m3docvqa_hyperlink_audit_mapped_full.edges.jsonl` |
| QID groups | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output/m3docvqa_qid_groups` |
| Doc-seed page-score ablation | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output/m3docvqa_doc_seed_page_score_ablation` |
| Adjacent-page edge ablation | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output/m3docvqa_gpp_edge_ablation_adj_w*` |
| Hyperlink-node MMR, target pages per doc 4 | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output/m3docvqa_gpp_hyperlink_node_ablation_mmr` |
| Hyperlink-node MMR, target pages per doc 1 | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output/m3docvqa_gpp_hyperlink_node_ablation_mmr_target1` |
| Hyperlink effect audit | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output/m3docvqa_hyperlink_init_ablation/log_count_hyperlink_effect_audit.md` |
| Dev-split doc fusion probe script | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/examples/run_m3docvqa_doc_fusion_devsplit_probe.sh` |

## Train/Dev Label Audit

| Split | rows | rows with qid | rows with gold docs | rows with page idx gold | rows with metadata page uid gold | Conclusion |
|---|---:|---:|---:|---:|---:|---|
| `MMQA_train.jsonl` | 23,817 | 23,817 | 23,817 | 0 | 0 | Train can support doc-level tuning, not true page-level supervised tuning. |
| `MMQA_dev.jsonl` | 2,441 | 2,441 | 2,441 | 0 | 0 | Dev can support doc-level split probes and grouped analysis. |

The gold `supporting_context` rows contain document ids and doc parts such as `table`, but no exact `page_idx`.

## Hyperlink File Audit

| Item | Count |
|---|---:|
| hyperlink edge rows | 21,451 |
| valid edge rows | 21,451 |
| source pages with hyperlinks | 13,451 |
| source docs with hyperlinks | 2,885 |
| target docs | 2,417 |
| gold qids | 2,441 |
| gold docs | 3,366 |
| gold docs with an incoming hyperlink | 2,417 |
| gold qids with any gold target inlink | 1,959 |
| gold qids with any gold source doc | 2,180 |
| qids with any retrieved source-page edge | 2,441 |
| qids with selected-doc hyperlink pairs | 2,350 |
| qids with retrieved link to a gold doc | 1,936 |

Interpretation:

- Hyperlinks are page-sourced: each edge row starts from one PDF page, for example `source_page_uid=..._page19`.
- The current mapped M3DocVQA edge file has target documents, not reliable target pages.
- Page-node hyperlink runs infer target pages by taking top retrieved candidate pages inside the target document. That is why `PDF_HYPERLINK_TARGET_PAGES_PER_DOC` matters.

## Ablation 1: Doc-Seed Page Score

Output folder: `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output/m3docvqa_doc_seed_page_score_ablation`

| Variant | doc@1 | doc@4 | doc@20 | row@4 | row@20 | Finding |
|---|---:|---:|---:|---:|---:|---|
| `docseed_none` | 0.608 | 0.846 | 0.927 | 0.758 | 0.893 | Baseline. |
| `docseed_rrf_0p50` | 0.608 | 0.842 | 0.926 | 0.761 | 0.894 | Row@4 rises, doc@4 drops. |
| `docseed_page_max_0p50` | 0.607 | 0.843 | 0.925 | 0.761 | 0.894 | Worse doc@4 than baseline. |
| `docseed_page_mean_0p50` | 0.607 | 0.844 | 0.927 | 0.760 | 0.893 | No useful gain. |
| `docseed_page_top3mean_0p50` | 0.608 | 0.844 | 0.926 | 0.760 | 0.894 | No useful gain. |
| `docseed_page_sum_0p10` | 0.608 | 0.847 | 0.927 | 0.757 | 0.892 | Tiny doc@4 gain, row@4 drops. |
| `docseed_page_sum_0p25` | 0.609 | 0.846 | 0.927 | 0.757 | 0.891 | No meaningful gain. |
| `docseed_page_sum_0p50` | 0.609 | 0.847 | 0.927 | 0.756 | 0.891 | Tiny doc@4 gain, row@4 drops. |

Grouped finding:

| Group | Best-looking variant | Improved | Worsened | Net | delta doc@4 | delta row@4 | Conclusion |
|---|---|---:|---:|---:|---:|---:|---|
| single gold doc | `docseed_page_sum_0p25` / `0p50` | 9 / 15 | 6 / 12 | +3 / +3 | 0.0000 | 0.0000 | Movement is small and not a useful main result. |
| multi gold doc | `docseed_page_sum_0p50` | 10 | 10 | 0 | +0.0005 | -0.0033 | Does not improve row context. |

Conclusion: do not use doc-seed page-score as a main M3DocVQA setting. It is weaker than hyperlink-node propagation.

## Ablation 2: Previous Hyperlink Initialization

Output folder: `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output/m3docvqa_hyperlink_init_ablation`

| Variant | doc@4 | doc@20 | row@4 | row@20 | Finding |
|---|---:|---:|---:|---:|---|
| no hyperlink MMR | 0.846 | 0.927 | 0.788 | 0.893 | Baseline for previous init method. |
| `log_count` | 0.851 | 0.930 | 0.798 | 0.898 | Best balanced default. |
| `sqrt_count` | 0.851 | 0.930 | 0.799 | 0.898 | Similar to `log_count`, slightly higher row@4. |
| `raw_count` | lower than log/sqrt in balance | lower than log/sqrt | lower balance | lower balance | Repeated links are too strong. |
| `uniform` | no important added value | no important added value | no important added value | no important added value | Ignores repeated-link evidence. |

Definitions:

- `uniform`: every hyperlink edge gets the same weight.
- `raw_count`: edge weight is proportional to repeated raw link count.
- `sqrt_count`: edge weight is `sqrt(raw_link_count)`, which dampens repeated links.
- `log_count`: edge weight is `log1p(raw_link_count)`, which dampens repeated links more aggressively.

Conclusion: keep `log_count` as the default hyperlink edge weighting. `sqrt_count` is competitive, but `log_count` is the safer balanced setting.

## Ablation 3: Adjacent Page Edge Weight

Output folders: `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output/m3docvqa_gpp_edge_ablation_adj_w*`

| Adjacent weight | overall doc@1 | overall doc@4 | overall doc@20 | single delta doc@4 | single delta row@4 | multi delta doc@4 | multi delta row@4 | Finding |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.00 | 0.860 | 0.960 | 0.984 | baseline | baseline | baseline | baseline | Baseline for this specific graph-summary run. |
| 0.10 | 0.859 | 0.960 | 0.984 | 0.0000 | 0.0000 | +0.0001 | -0.0007 | No meaningful change. |
| 0.25 | 0.858 | 0.961 | 0.984 | +0.0027 | -0.0009 | +0.0004 | -0.0038 | Small doc@4 gain, row cost. |
| 0.50 | 0.858 | 0.961 | 0.984 | +0.0027 | -0.0018 | -0.0001 | -0.0054 | Row cost grows. |
| 1.00 | 0.856 | 0.962 | 0.984 | +0.0055 | -0.0009 | -0.0003 | -0.0063 | Best overall doc@4, but worse top-1 and row context. |

Conclusion: adjacent-page edges are not a strong M3DocVQA feature. They can slightly improve document recall at higher weights, but they hurt row context for multi-gold-doc qids. Keep small/default adjacent weight only as a background structural edge, not as a headline ablation.

## Ablation 4: Current GPP Hyperlink Node Mode With MMR

Output folder: `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output/m3docvqa_gpp_hyperlink_node_ablation_mmr`

Config:

```bash
FINAL_SELECTION_MODE=mmr_doc_diverse
FINAL_SELECTION_NEW_DOC_BONUS=0.10
FINAL_SELECTION_SAME_DOC_PENALTY=0.05
PDF_HYPERLINK_TARGET_PAGES_PER_DOC=4
```

Overall table:

| Variant | doc@1 | doc@2 | doc@4 | doc@10 | doc@20 | doc@50 | doc@100 | row@4 | row@10 | row@20 | row@50 | row@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no hyperlink | 0.608 | 0.768 | 0.846 | 0.903 | 0.927 | 0.948 | 0.963 | 0.788 | 0.849 | 0.893 | 0.930 | 0.946 |
| `docnode_to_hyperlink_docs` | 0.608 | 0.772 | 0.851 | 0.907 | 0.930 | 0.948 | 0.963 | 0.798 | 0.855 | 0.898 | 0.932 | 0.946 |
| `pagenode_to_hyperlink_pages` | 0.604 | 0.775 | 0.854 | 0.908 | 0.933 | 0.958 | 0.975 | 0.768 | 0.857 | 0.899 | 0.938 | 0.954 |

Grouped table:

| Group | Variant | improved | worsened | net | doc@4 | row@4 | delta doc@4 | delta row@4 | Finding |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| single gold doc | no hyperlink | - | - | - | 0.9443 | 0.9288 | - | - | Baseline. |
| single gold doc | `docnode_to_hyperlink_docs` | 28 | 14 | +14 | 0.9471 | 0.9325 | +0.0027 | +0.0036 | Conservative gain. |
| single gold doc | `pagenode_to_hyperlink_pages` | 53 | 35 | +18 | 0.9507 | 0.9325 | +0.0064 | +0.0036 | Best single-doc doc@4. |
| multi gold doc | no hyperlink | - | - | - | 0.7667 | 0.6724 | - | - | Baseline. |
| multi gold doc | `docnode_to_hyperlink_docs` | 12 | 23 | -11 | 0.7730 | 0.6893 | +0.0063 | +0.0169 | Recall rises even though movement net is negative. |
| multi gold doc | `pagenode_to_hyperlink_pages` | 35 | 43 | -8 | 0.7743 | 0.6899 | +0.0077 | +0.0175 | Best multi-doc recall/context proxy. |

Interpretation:

- `docnode_to_hyperlink_docs` connects retrieved document nodes to hyperlink target document nodes.
- `pagenode_to_hyperlink_pages` connects retrieved page nodes to inferred pages from hyperlink target documents.
- Since the hyperlink edge file does not contain exact target pages, `pagenode_to_hyperlink_pages` is approximate. It uses top retrieved pages from the target document.
- The negative multi-doc `net` is not a contradiction: the method can worsen the first gold-hit movement for more qids while still increasing total gold-doc coverage in top-k.

## Ablation 5: Target Pages Per Hyperlink Target Doc

Output folder for target-1 run: `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output/m3docvqa_gpp_hyperlink_node_ablation_mmr_target1`

This tests the approximation used by `pagenode_to_hyperlink_pages`.

| Variant | target pages per doc | doc@1 | doc@4 | doc@10 | doc@20 | doc@50 | doc@100 | row@1 | row@4 | row@10 | row@20 | row@50 | row@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no hyperlink MMR | 0 | 0.608 | 0.846 | 0.903 | 0.927 | 0.948 | 0.963 | 0.608 | 0.788 | 0.849 | 0.893 | 0.930 | 0.946 |
| `docnode_to_hyperlink_docs` | n/a | 0.608 | 0.851 | 0.907 | 0.930 | 0.948 | 0.963 | 0.608 | 0.798 | 0.855 | 0.898 | 0.932 | 0.946 |
| `pagenode_to_hyperlink_pages` | 4 | 0.604 | 0.854 | 0.908 | 0.933 | 0.958 | 0.975 | 0.604 | 0.768 | 0.857 | 0.899 | 0.938 | 0.954 |
| `pagenode_to_hyperlink_pages` | 1 | 0.600 | 0.853 | 0.911 | 0.936 | 0.961 | 0.975 | 0.600 | 0.804 | 0.864 | 0.905 | 0.942 | 0.958 |

Conclusion:

- `target_pages_per_doc=1` is the best current pagenode setting for returned context and deeper document recall.
- `target_pages_per_doc=4` is slightly better for `doc@4` (`0.854` vs `0.853`) but much worse for `row@4` (`0.768` vs `0.804`).
- Use target-1 as the current default unless the report optimizes only `doc@4`.

## Ablation 6: External Dataset Hyperlink Applicability

The M3DocVQA hyperlink approach depends on mapping extracted PDF links to dataset-internal target docs. The external dataset sanity checks show that this mapping is not available yet.

| Dataset | local PDF count | sanity subset | hyperlink records | raw Wikipedia article URLs | mapped valid internal edges | Conclusion |
|---|---:|---:|---:|---:|---:|---|
| M3DocVQA/MMQA | 27,528 total, 3,366 dev | full dev hyperlink file | 21,451 mapped edges | mapped already | 21,451 | Usable now. |
| OpenDocVQA | not in prepared PDF form for the checked setup | first 20 docs | 0 | 0 | 0 | No usable hyperlink graph. |
| ViDoRe V3 | checked prepared pages | first 20 docs | 492 noisy OCR URLs | 0 mapped | 0 | No usable internal link graph. |
| DUDE | 5,075 | first 20 docs | 137 | 9 | 0 | PDFs have links, but no internal target mapping. |
| ViDoSeek | 292 | first 20 docs | 61 | 8 | 0 | PDFs have links, but no internal target mapping. |
| SciEGQA | 80 | first 20 docs | 2,336 | 1 | 0 | Many PDF links, but no internal target mapping. |
| MMDocIR | 0 local PDFs in prepared data | n/a | n/a | n/a | 0 | Cannot extract PDF hyperlinks from prepared data. |
| MMLongBench DocQA | 0 local PDFs in prepared data | n/a | n/a | n/a | 0 | Image-only prepared data. |

Conclusion: hyperlink-node GPP is currently a M3DocVQA-only method. For OpenDocVQA/ViDoRe/DUDE/ViDoSeek/SciEGQA, we would need a dataset-specific URL-to-doc mapping before running the same graph edge family.

## Ablation 7: Page Position Priors Status

Implemented pieces:

- Same-document neighboring pages through `same_doc_window` and `adjacent_page_edge_weight`.
- Query-gated position evidence in `scripts/graph_rerank_page_retrieval_predictions.py` for first/cover/early pages, TOC-like cues, last/final/late pages, references, appendix, acknowledgements, and explicit page/slide numbers.
- Standalone structural metadata reranker in `scripts/rerank_structural_metadata_pages.py`.

Not yet implemented as a main M3DocVQA run:

- A learned page-position prior.
- A generic table-page prior.
- A wrapper-level M3DocVQA sweep exposing `POSITION_EVIDENCE_*` flags.

Conclusion: position priors exist as partial/query-gated features, but today's M3DocVQA ablations did not establish them as a main result.

## Dev-Split Doc Fusion Probe

Implemented but not yet reported from a user-run result:

| Item | Path |
|---|---|
| Script | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/scripts/tune_m3docvqa_doc_fusion_split.py` |
| Runner | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/examples/run_m3docvqa_doc_fusion_devsplit_probe.sh` |
| Expected output dir | `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output/m3docvqa_doc_fusion_devsplit_probe` |

Recommended run:

```bash
cd /mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG
git pull --rebase origin codex/mmdocir-hpc-workflow
source hpc_vital_paths.generated.env
bash examples/run_m3docvqa_doc_fusion_devsplit_probe.sh
```

Use this as the green-light check before doing train-set doc-level tuning. Because train has gold docs but no gold pages, this should tune document-level fusion/graph weights only.

## Final Conclusion

The strongest finding today is not doc-seed scoring or adjacent-page weighting. It is hyperlink propagation inside GPP, especially when combined with MMR document-diverse final selection.

Current reporting hierarchy:

1. Main M3DocVQA GPP hyperlink result: `pagenode_to_hyperlink_pages + log_count + target_pages_per_doc=1 + MMR doc-diverse`.
2. Conservative alternative: `docnode_to_hyperlink_docs + log_count + MMR doc-diverse`, because it preserves top-1 while improving doc@4/row@4.
3. Historical comparison: previous hyperlink initialization with `log_count`/`sqrt_count` improves over no hyperlink, but the newer node-level pagenode target-1 setting is better for row/context.
4. Negative/weak findings: doc-seed page-score, raw/uniform hyperlink weighting, high adjacent-page weights, and external dataset hyperlinks without internal mapping.
