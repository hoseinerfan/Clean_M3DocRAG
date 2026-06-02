# Cross-Dataset Retrieval Scoreboard

Purpose: keep one short scoreboard for the non-MMQA page-labeled datasets. Unlike MMQA, these datasets are currently retrieval-only in this repo context, so the `Table C` sections below report page/doc retrieval metrics only.

Primary source note:

- [notes/graph_ppr_external_datasets_handoff_2026-05-21.md](/Users/hoseinerfan/Desktop/Clean_M3DocRAG/notes/graph_ppr_external_datasets_handoff_2026-05-21.md:1)
- [notes/graph_ablation_findings_2026-05-28.md](/Users/hoseinerfan/Desktop/Clean_M3DocRAG/notes/graph_ablation_findings_2026-05-28.md:1)

## Metric Policy

- For page-labeled datasets, report:
  - `page@1`
  - `page@4`
  - `page@20`
  - `doc@4`
  - `doc@20`
- Keep `page@1` separate from `page@4/page@20`, because rank-1 behavior can differ from broader page retrieval quality.
- Use `plain_top224` as the dense baseline comparator.
- Use page-preserving Graph-PPR rows as the graph comparators.

## Current Default

- strongest frozen single general Graph-PPR config for page-labeled datasets:
  - `denseheavy125_medium_both`
- exception:
  - `ViDoSeek` best individual row is currently `denseheavy150_m3best_pagepreserve`

## Cross-Dataset Summary

| Dataset | Best Graph Row | qids | page@1 | page@4 | Conservative gate page hit@4 | page@20 | doc@4 | doc@20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SciEGQA | `denseheavy125_medium_both` | `1,623` | `0.5508` *(plain `0.5228`)* | `0.8152` *(plain `0.7394`)* | `0.8152` (`1323 / 1623`; native, no net change) | `0.9248` *(plain `0.8758`)* | `0.9291` *(plain `0.9070`)* | `0.9871` *(plain `0.9772`)* |
| MMDocIR | `denseheavy125_medium_both` | `1,658` | `0.4596` *(plain `0.4136`)* | `0.6719` *(plain `0.6075`)* | `0.6737` (`1117 / 1658`; `+3`, `0` lost) | `0.7889` *(plain `0.7480`)* | `0.8160` *(plain `0.8058`)* | `0.8938` *(plain `0.8890`)* |
| ViDoRe V3 | `denseheavy125_medium_both` | `14,514` | `0.3902` *(plain `0.1730`)* | `0.6465` *(plain `0.3312`)* | `0.6465` (`9383 / 14514`; heading no-op) | `0.8227` *(plain `0.5431`)* | `0.9099` *(plain `0.8854`)* | `0.9788` *(plain `0.9809`)* |
| ViDoSeek | `denseheavy150_m3best_pagepreserve` | `1,142` | `0.6909` *(plain `0.6830`)* | `0.9037` *(plain `0.8958`)* | `0.9011`* (`1029 / 1142`; `+6`, `0` lost) | `0.9982` *(plain `0.9842`)* | `0.9991` *(plain `0.9982`)* | `1.0000` *(plain `1.0000`)* |

The conservative-gate column reports measured final top-4 page-hit output after native
heading/bodyguard rescue. It is limited to `@4` because the gate is a rank-boundary rescue
layer. `*` The ViDoSeek result starts from its no-heading control branch (`1023 -> 1029`
hits), rather than directly composing the gate with the separately selected
`denseheavy150_m3best_pagepreserve` row.

## Trained Content-Aware Transfer 2026-06-02

Detailed interpretation note: [page_evidence_promotion_findings_2026-06-02.md](/Users/hoseinerfan/Desktop/Clean_M3DocRAG/notes/page_evidence_promotion_findings_2026-06-02.md:1)

The M3DocVQA/MMQA-trained adaptive content-aware page promotion model was applied zero-shot to DUDE, MMDocIR, SciEGQA, and ViDoSeek. It used the saved M3DocVQA model weights and `blend_alpha` from the adaptive `page@5` run; no target-dataset labels were used for training.

Dense-base transfer:

| Dataset | dense page@5 | transfer page@5 | page@5 gain | page@4 gain | page@10 gain | doc@4 gain | Finding |
|---|---:|---:|---:|---:|---:|---:|---|
| ViDoSeek | 0.9151 | 0.9229 | +0.0079 | +0.0070 | +0.0053 | +0.0000 | Small positive; dataset is already saturated. |
| SciEGQA | 0.8065 | 0.8226 | +0.0160 | +0.0080 | +0.0117 | +0.0105 | Positive transfer. |
| DUDE | 0.5908 | 0.6173 | +0.0265 | +0.0269 | +0.0200 | +0.0107 | Strongest positive transfer. |
| MMDocIR | 0.6580 | 0.6821 | +0.0241 | +0.0151 | +0.0115 | +0.0018 | Positive transfer. |

Docseed-base transfer:

| Dataset | page@4 gain | page@5 gain | page@10 gain | doc@4 gain | Finding |
|---|---:|---:|---:|---:|---|
| ViDoSeek | +0.0061 | +0.0026 | +0.0053 | +0.0000 | Small positive. |
| SciEGQA | +0.0043 | +0.0111 | -0.0043 | +0.0018 | Mixed but acceptable at top-5. |
| DUDE | -0.0086 | -0.0183 | -0.0541 | -0.0241 | Negative transfer after docseed. |
| MMDocIR | -0.0320 | -0.0326 | -0.0513 | -0.0247 | Negative transfer after docseed. |

Current transfer conclusion:

- The M3DocVQA-trained content-aware model improves dense retrieval zero-shot on all four checked datasets.
- It is not a universal post-reranker after stronger docseed outputs.
- For DUDE and MMDocIR, `docseed_rrf_1p00` remains stronger than `docseed_rrf_1p00 + transfer`.
- Report the method as a dense-pool evidence promotion transfer, not as a replacement for every dataset-specific graph/docseed pipeline.

## Table C: SciEGQA

| Method | qids | page@1 | page@4 | page@20 | doc@4 | doc@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `plain_top224` | `1,623` | `0.5228` | `0.7394` | `0.8758` | `0.9070` | `0.9772` |
| `Graph Page Preserve` (`denseheavy125_medium_both`) | `1,623` | `0.5508` | `0.8152` | `0.9248` | `0.9291` | `0.9871` |

Reading:

- broad win for the page-preserving graph row
- page@4, page@20, doc@4, and doc@20 all improve over `plain_top224`
- this is the clearest external-dataset confirmation that the page-preserving fix works

Neighbor-page expansion check:

| Method | qids | page@1 | page@4 | page@20 | doc@4 | doc@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Graph Page Preserve` (`denseheavy125_medium_both`) | `1,623` | `0.4951` | `0.7733` | `0.9116` | `0.9291` | `0.9871` |
| `Graph Page Preserve + Neighbor1` (`denseheavy125_medium_both_neighbor1`) | `1,623` | `0.4969` | `0.7748` | `0.9187` | `0.9273` | `0.9864` |

Reading:

- `neighbor1` gives a very small page gain:
  - `page@1`: `+0.0018`
  - `page@4`: `+0.0015`
  - `page@20`: `+0.0071`
- but it slightly hurts document retrieval:
  - `doc@4`: `-0.0018`
  - `doc@20`: `-0.0006`
- current conclusion:
  - neighbor-page expansion is promising as a page-localization idea
  - but the naive `page±1` expansion is too small a win to replace `denseheavy125_medium_both`

## Table C: MMDocIR

| Method | qids | page@1 | page@4 | page@20 | doc@4 | doc@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `plain_top224` | `1,658` | `0.4136` | `0.6075` | `0.7480` | `0.8058` | `0.8890` |
| `Graph Page Preserve` (`denseheavy125_medium_both`) | `1,658` | `0.4596` | `0.6719` | `0.7889` | `0.8160` | `0.8938` |

Reading:

- clear page@1/@4/@20 win
- small but real doc@4/doc@20 gains
- this is a clean transfer win over `plain_top224`

## Table C: ViDoRe V3

| Method | qids | page@1 | page@4 | page@20 | doc@4 | doc@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `plain_top224` | `14,514` | `0.1730` | `0.3312` | `0.5431` | `0.8854` | `0.9809` |
| `Graph Page Preserve` (`denseheavy125_medium_both`) | `14,514` | `0.3902` | `0.6465` | `0.8227` | `0.9099` | `0.9788` |

Reading:

- large page retrieval gain at all practical depths
- doc@4 improves
- doc@20 drops slightly
- rank-1 and practical page retrieval both strongly favor the graph row here

## Table C: ViDoSeek

| Method | qids | page@1 | page@4 | page@20 | doc@4 | doc@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `plain_top224` | `1,142` | `0.6830` | `0.8958` | `0.9842` | `0.9982` | `1.0000` |
| `Graph Page Preserve` (`denseheavy150_m3best_pagepreserve`) | `1,142` | `0.6909` | `0.9037` | `0.9982` | `0.9991` | `1.0000` |

Reading:

- saturated dataset
- the gains are small but still positive at page@1/page@4/page@20
- doc@20 is already saturated for both methods
- use `denseheavy150_m3best_pagepreserve` if optimizing ViDoSeek alone

Important note:

- `denseheavy125_medium_both` is still the best **general** frozen config across datasets
- `denseheavy150_m3best_pagepreserve` is currently a ViDoSeek-specific best row

## Doc-Doc And Doc-Seed Ablation Status

Latest focused ablations on MMDocIR, SciEGQA, and ViDoSeek did not replace the page-labeled default.

| Dataset | main page-labeled setting after ablation | useful ablation exception | Reason |
|---|---|---|---|
| MMDocIR | `denseheavy125_medium_both`, no doc-doc edges, no doc seed | none | doc-doc edges can improve doc@4 but hurt page@4; doc seed hurts doc@4. |
| SciEGQA | `denseheavy125_medium_both`, no doc-doc edges | `docseed_rrf_1p00` | doc seed improves page@4 by `+12` and doc@4 by `+4`; doc-doc edges hurt. |
| ViDoSeek | page-preserving graph, no doc-doc edges, no doc seed | none | doc@4 is saturated and page@4 is unchanged or worse. |
| ViDoRe V3 | pending complete ablation | cross-doc selection likely relevant | `1,386` qids have gold pages across different docs. |
| OpenDocVQA | pending complete ablation | cross-doc selection likely relevant | `9,348` qids have gold pages across different docs. |

Mechanism audits:

- dense/sparse agreement edges passed strict audits.
- semantic-similarity edges were active but neutral on MMDocIR/SciEGQA/ViDoSeek.
- shared-entity/title/topic edges were inactive (`0` edge qids) on the checked datasets.

## Earlier Transfer Table That Failed

This is the old M3DocVQA-tuned `doc_shortlist_best` transfer that motivated the switch to page-preserving output. Values outside parentheses are `doc_shortlist_best`; values in parentheses are `plain_top224`.

| Dataset | qids | doc@1 | doc@4 | doc@20 | page@1 | page@4 | page@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MMDocIR | `1,658` | `0.6815` *(0.6852)* | `0.8034` *(0.8058)* | `0.8884` *(0.8890)* | `0.3861` *(0.4136)* | `0.4562` *(0.6075)* | `0.4998` *(0.7480)* |
| SciEGQA | `1,623` | `0.8355` *(0.8262)* | `0.9279` *(0.9070)* | `0.9846` *(0.9772)* | `0.4624` *(0.5228)* | `0.5173` *(0.7394)* | `0.5474` *(0.8758)* |
| ViDoSeek | `1,142` | `0.9860` *(0.9939)* | `0.9991` *(0.9982)* | `1.0000` *(1.0000)* | `0.6655` *(0.6830)* | `0.6743` *(0.8958)* | `0.6751` *(0.9842)* |
| ViDoRe V3 | `14,514` | `0.6521` *(0.6586)* | `0.8725` *(0.8854)* | `0.9703` *(0.9809)* | `0.1472` *(0.1730)* | `0.2064` *(0.3312)* | `0.2285` *(0.5431)* |

Why keep this table:

- it explains why the repo moved away from one-page-per-doc graph output for page-labeled benchmarks
- it is the negative-transfer baseline that the new page-preserving tables above should be compared against conceptually

## Extension Template

When adding a new dataset, keep the same pattern:

1. one summary row in the cross-dataset summary table
2. one dataset-specific `Table C`
3. one short reading block
4. explicit statement of:
   - dense baseline row
   - best graph row
   - whether the best row is dataset-specific or part of the general default
