# Graph-PPR Novelty Tracking

This note tracks the current novelty story, empirical status, and next experiments for query-adaptive graph retrieval. The goal is to separate methods with real thesis novelty from practical heuristics, and to keep the next runs focused on non-label, non-oracle improvements.

Latest ablation findings: [graph_ablation_findings_2026-05-28.md](/Users/hoseinerfan/Desktop/Clean_M3DocRAG/notes/graph_ablation_findings_2026-05-28.md:1)

Latest page-evidence promotion findings: [page_evidence_promotion_findings_2026-06-02.md](/Users/hoseinerfan/Desktop/Clean_M3DocRAG/notes/page_evidence_promotion_findings_2026-06-02.md:1)

## Working Novelty Claim

We propose a query-adaptive heterogeneous evidence graph for multimodal multi-page document retrieval. Dense and sparse candidate pages are augmented with typed evidence nodes derived from query-specific retrieval reliability, document structure, page position, and query anchors. Personalized PageRank is then used to propagate evidence over this graph.

This is stronger than simply using ColPali, SPLADE, RRF, or standard PPR. Those are existing tools. The novelty should be claimed in the graph construction and query-adaptive evidence design.

## Related Page-Promotion Track

The pseudo-page-supervised page-promotion work is now a separate but connected thesis track. It uses graph-derived outputs as features, but it does not claim novelty as a new graph propagation algorithm.

Current status:

| Method | Current result | Verdict |
|---|---:|---|
| Content-aware promotion, adaptive `page@5` | M3DocVQA strict pseudo-page `page@5=0.8031` | Best current page-evidence method. |
| Counterfactual page promotion, insert rank 5 | M3DocVQA strict pseudo-page `page@5=0.7707`, gain `+0.0486` over GPP no-hyperlink | Safe repair extension; preserves top-4. |
| M3DocVQA-trained content transfer on dense pools | positive zero-shot `page@5` gains on ViDoSeek, SciEGQA, DUDE, and MMDocIR | Useful transfer result, but not a universal post-reranker. |
| LightGBM LambdaMART | improves `page@4` over GPP no-hyperlink but hurts broader ranking | Control/extension, not main method. |

Advisor-facing separation:

- Graph-PPR novelty: query-adaptive graph construction and typed evidence propagation.
- Page-promotion novelty: pseudo-page supervision, discovery-vs-promotion diagnosis, and graph-aware content evidence promotion.
- Generic LTR and fixed heuristic promotion are controls, not the central novelty claims.

## Current Performance Ledger

All MMDocIR numbers below are page hit@4 unless otherwise noted.

| Method | Best observed MMDocIR result | Delta vs graph baseline | Current verdict |
| --- | ---: | ---: | --- |
| Graph-PPR baseline, denseheavy125 medium both | 1114 / 1658, 67.19% | baseline | Anchor result |
| Reciprocal source reliability weighting | 1115 / 1658 | +1 | Small, stable gain; good non-oracle reliability idea |
| Reliability-aware restart | 1109 / 1658 in original adaptive restart; page-only variants reached 1114 | -5 to 0 | Not useful yet |
| Adaptive transition / asymmetric page-doc propagation | about 1113-1114 / 1658 | -1 to 0 | Mostly neutral |
| Coherence-gated same-doc edges | 1115 / 1658 | +1 | Small gain; low-risk structural graph feature |
| Evidence community nodes | 1111-1114 / 1658 | -3 to 0 | Not useful yet |
| Query-position graph nodes | 1114 / 1658 | 0 | Conceptually useful but empirically neutral so far |
| Heading/breadcrumb anchor nodes | pending | pending | Graph-native section bridge for query-named headings; implemented, needs ablation |
| Entity/alias anchor nodes | pending | pending | Corpus entity graph with alias normalization; implemented, needs markdown-backed ablation |
| Structural metadata reranker, conservative parser | 1118 / 1658 | +4, lost=0 | Best targeted gain, but rule-based/heuristic |
| Query anchor evidence nodes, uniform w0.20 r0.05 | 1117 / 1658 | +3, lost=0 | Best graph-native novelty result so far |
| Query anchor r0.10 + tight structural metadata | 1123 / 1658 | +9, lost=0 | Best current MMDocIR result, but includes heuristic structural layer |
| Financial verifier v2, broad table/text evidence | 1120 / 1658 | +6, lost=0 vs denseheavy baseline in case compare | Useful diagnostic; evidence often too broad |
| Financial verifier strict doc-prior | 1118 / 1658 | +4, but lost=2 | Reduced broad positives, but too strict for headline use |
| SPLADE/BM25 page-page kNN graph | SPLADE implemented; BM25 implemented, pending run | pending BM25 | Retriever-induced semantic/lexical graph views with mutual-kNN noise control |
| LayoutLMv3 fallback-text kNN graph | 1104-1114 / 1658 | -10 to 0 | Negative ablation until real OCR/layout boxes are available |

M3DocVQA/MMQA document retrieval numbers:

| Method | doc@1 | doc@4 | row@4 | row@10 | Current verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| no-hyperlink Graph Page Preserve | 0.608 | 0.846 | 0.758 | 0.848 | baseline |
| PDF hyperlink graph default `w0.10` | 0.608 | 0.847 | 0.759 | 0.849 | small gain in complete wrapper |
| PDF hyperlink graph tuned `w2.25` | 0.608 | 0.851 | 0.770 | 0.854 | best balanced hyperlink setting |
| PDF hyperlink graph `w5.00` | 0.605 | 0.850 | 0.769 | 0.856 | stronger row@10, starts hurting doc@1 |
| MMR doc-diverse final selection `b0.10` | 0.608 | 0.846 | 0.799 | 0.850 | best row@4 selection gain, doc@4 neutral |

Cross-dataset structural metadata sanity:

| Dataset | Result |
| --- | --- |
| MMDocIR | +4 page_hit@4, lost=0 |
| SciEGQA | parser-tightened version is neutral, lost=0 |

## What Counts As Novel

Strongest thesis-facing novelty:

1. Reciprocal source reliability estimation from dense/SPLADE cross-support.
2. Query-adaptive graph propagation controlled by reliability, transition, and same-document coherence.
3. Typed evidence nodes in a heterogeneous graph: query-position nodes, query-anchor nodes, heading/breadcrumb nodes, entity/alias nodes, and planned page-type/visual-tag nodes.
4. Failure-driven graph augmentation: target cases where the gold page exists in the dense/sparse pool but graph ranking fails to localize it.
5. Authored hyperlink graph augmentation for Wikipedia-derived PDF corpora: PDF annotation links create real document/document transitions rather than synthetic similarity edges.

Not novel by itself:

1. ColPali or dense retrieval.
2. SPLADE retrieval.
3. RRF.
4. Standard Personalized PageRank.
5. Simple post-hoc boosting.
6. Hand-written query rules without graph integration.
7. Fallback text-only LayoutLMv3 kNN edges; without real OCR boxes/layout features this is not a layout graph.

## Heuristic Status

The structural metadata reranker is useful but heuristic. It should be presented as a targeted, interpretable module or ablation, not as the main thesis novelty.

The graph-native methods are the main novelty path. They should be prioritized when looking for advisor-facing contributions:

1. Query anchor evidence nodes.
2. SPLADE/BM25 retriever-induced page-page graph views.
3. Heading/breadcrumb anchor nodes.
4. Entity/alias anchor nodes.
5. Page-type/modality nodes.
6. Section-role nodes.
7. Visual tag nodes from VLM/captioning.
8. Reliability-weighted propagation and source weighting.

## Next Experiments

### 0. Constraint-Aware Evidence Bundle Nodes

Purpose: target the clearest current graph limitation: Graph-PPR can identify the right document but still localizes the wrong page inside that document.

Current implementation:

- Enables `QUERY_ANCHOR_REASONING_MODE=constraint_bundles` inside `scripts/graph_rerank_page_retrieval_predictions.py`.
- Extracts query-side constraint slots:
  - `entity`
  - `numeric`
  - `metric`
  - `role`
- Adds one conjunctive graph node when candidate pages satisfy multiple constraint types together.
- Requires critical `entity` and `numeric` slots when they exist, so pages matching only broad metric/year/table evidence do not get the bundle boost.
- Downweights broad slots by local specificity and optionally drops overly broad slots or bundles.

Why this is novel:

- The graph no longer propagates relevance only through page/doc nodes or independent lexical anchors.
- It creates a query-conditioned typed evidence bottleneck, so document-level PPR mass can return preferentially to pages satisfying the query's evidence constraints.
- This directly tests whether the graph can solve right-document/wrong-page failures without adding another retriever or VLM.

Recommended first settings:

```bash
QUERY_ANCHOR_EVIDENCE_MODE=entity_numeric
QUERY_ANCHOR_REASONING_MODE=constraint_bundles
QUERY_ANCHOR_SCOPE=doc_conditioned
QUERY_ANCHOR_DOC_TOP_K=20
QUERY_ANCHOR_EDGE_WEIGHT=0.15
QUERY_ANCHOR_RESTART_WEIGHT=0.10
QUERY_ANCHOR_CONSTRAINT_BUNDLE_WEIGHT=1.0
QUERY_ANCHOR_CONSTRAINT_MIN_SLOT_TYPES=2
QUERY_ANCHOR_CONSTRAINT_SPECIFICITY_FLOOR=0.10
```

Primary target bucket:

- page@4 misses where doc@4 already hits
- especially MMDocIR metadata/table-like queries and ViDoRe V3 finance/table failures

Success criteria:

- Recover page@4 misses with low or zero loss against `denseheavy125_medium_both`.
- Summary JSON should show nonzero `query_anchor_constraint_bundle_qid_count`.
- Audit recovered/lost cases to verify the bundle is selecting answer-bearing evidence pages, not just matching broad table text.

### 1. Query Anchor Evidence Nodes

Purpose: improve entity/numeric localization in finance, news, and papers without gold labels.

Current implementation:

- Extracts entity-like and numeric anchors from the query.
- Reads page text from `doc_pages_dev.jsonl`.
- Adds `query_anchor::*` nodes.
- Connects anchor nodes to candidate pages containing the anchor.
- Supports doc-conditioned matching to avoid global false positives.
- Supports query-local IDF/specificity weighting to downweight broad anchors.

Current result:

```text
Graph baseline: 1114 page_hit@4, 1353 doc_hit@4
Query-anchor uniform w0.20 r0.05: 1117 page_hit@4, 1356 doc_hit@4
Recovered/lost: 3 / 0
Query-anchor uniform w0.20 r0.10: 1119 page_hit@4, 1358 doc_hit@4
Recovered/lost: 5 / 0
Improved/worsened ranks: 187 / 113
Query-anchor r0.10 + tight structural metadata: 1123 page_hit@4, 1358 doc_hit@4
Recovered/lost: 9 / 0
Improved/worsened ranks: 183 / 127
```

Audit observations:

```text
active qids: 1210 / 1658
gold-anchor match among active qids: 1120 / 1210
mean page matches per active qid: 392
active improved/worsened ranks: 115 / 67
```

This confirms the mechanism is meaningful. Local-IDF/cap variants did not improve over
uniform weighting, so the current best graph-native candidate is uniform query-anchor
evidence with `QUERY_ANCHOR_EDGE_WEIGHT=0.20` and `QUERY_ANCHOR_RESTART_WEIGHT=0.10`.
The best full system currently combines that graph-native query-anchor run with the
tight structural metadata reranker.

First runs:

```bash
QUERY_ANCHOR_EVIDENCE_MODE=entity_numeric
QUERY_ANCHOR_SCOPE=doc_conditioned
QUERY_ANCHOR_DOC_TOP_K=20
QUERY_ANCHOR_EDGE_WEIGHT=0.10
QUERY_ANCHOR_RESTART_WEIGHT=0.05
```

Selectivity runs:

```bash
QUERY_ANCHOR_WEIGHT_MODE=local_idf
QUERY_ANCHOR_MIN_NODE_WEIGHT=0.10
QUERY_ANCHOR_MAX_PAGE_MATCHES=0
```

and, if broad anchors remain noisy:

```bash
QUERY_ANCHOR_WEIGHT_MODE=local_idf
QUERY_ANCHOR_MAX_PAGE_MATCHES=500
```

Success criteria:

- Overall MMDocIR page_hit@4 improves by at least +3 with no large losses.
- Or targeted gains appear in `Financial report`, `News`, or `Academic paper`.
- Summary must show nonzero `doc_page_text_page_count` and nonzero `query_anchor_evidence_qid_count`.

If neutral:

- Try edge-only: `QUERY_ANCHOR_RESTART_WEIGHT=0.0`.
- Try stronger edge: `QUERY_ANCHOR_EDGE_WEIGHT=0.20`.
- Try top-doc restriction: `QUERY_ANCHOR_DOC_TOP_K=4` or `10`.
- Audit recovered/lost qids by domain and metadata type.

Financial audit from the current best query-anchor run:

```text
Financial report qids: 344
Recovered/lost: 2 / 0
Improved/worsened ranks: 35 / 29
Active financial qids: 303
Gold-anchor match among active financial qids: 280
```

Next financial-specific test:

- Add `QUERY_ANCHOR_REASONING_MODE=financial_slots`.
- Keep the same query-anchor edge/restart setting.
- Reward pages only when multiple financial evidence slots co-occur, e.g. metric + year,
  metric + entity, or multiple metric terms.
- This is a retrieval-time reasoning signal, not answer leakage: it uses only the query,
  retrieved candidate docs/pages, and corpus page text.

First financial-slot run was neutral overall: it matched `anchor_r010` on page/doc hit.
Audit showed the layer activated on 128 / 344 financial qids with mean 45 bundle pages,
but the slot extractor missed lowercase metric phrases and still included broad bundles.
The next revision adds lowercase financial metric phrase extraction and optional
numeric/table-likeness weighting via `QUERY_ANCHOR_FINANCIAL_TABLE_BONUS`.

Latest financial retrieval audit:

```text
Financial qids: 344
Broad financial verifier:
  recovered: 3
  improved_rank: 49
  worsened_rank: 29
  missing_in_both: 18
  main limitations:
    gold_page_no_financial_evidence_match: 187
    evidence_too_broad: 87
    gold_page_has_evidence_but_not_in_candidate_head: 29

Strict doc-prior verifier:
  recovered: 3
  lost: 2
  improved_rank: 51
  worsened_rank: 46
  active_qid_count: 167
  mean_active_positive_page_count: 76.24
```

Conclusion:

- broad verifier catches real evidence but matches too many pages
- strict verifier reduces broad positives but can demote already-correct top-4 pages
- do not make this the main method yet
- use it as evidence for the need for learned/query-conditioned evidence calibration

### 2. Page-Type / Modality Nodes

Purpose: help table/chart/figure failures by adding page-level type evidence.

Candidate design:

- Add nodes like `page_type::table`, `page_type::figure`, `page_type::chart`, `page_type::text`.
- Connect pages to page-type nodes using corpus-side page annotations or a page classifier.
- Activate/query-seed page-type nodes only when the query asks for a table, chart, figure, plot, graph, diagram, image, code block, signature, or map.

This is more thesis-worthy if page types are produced by a reusable model or VLM/captioning pipeline rather than hand labels.

### 3. Section-Role Nodes

Purpose: help metadata and navigation queries such as references, appendix, signatures, cover, final page, methods, results, tables of contents.

Candidate design:

- Add `section_role::references`, `section_role::appendix`, `section_role::toc`, `section_role::cover`, `section_role::signature`, `section_role::financial_table`.
- Derive roles from page text, headings, layout cues, and page position.
- Query activates role nodes through intent extraction.

This overlaps with structural metadata, but graph integration is a cleaner novelty path.

### 4. Visual Tag Nodes

Purpose: highest novelty for M3DocVQA/ImageListQ and visual questions.

Candidate design:

- Generate page-level VLM tags/captions offline.
- Add nodes for salient visual entities and attributes: logo, person, map, chart, table, signature, trophy, color, animal, vehicle, etc.
- Connect query visual anchors to page visual tags.

This is likely the most work but the strongest new contribution if performance improves.

### 5. Reliability-Aware Propagation Refinement

Purpose: keep the non-heuristic reliability story alive.

Candidate design:

- Use reciprocal support to adapt source weights, restart vector, and edge strengths.
- Avoid fixed domain/type routing unless used only for analysis.
- Evaluate by agreement bins, pool coverage, and source-only gold coverage.

### 6. PDF Hyperlink Graph For M3DocVQA/MMQA

Purpose: use authored Wikipedia links embedded in the PDF annotations to connect retrieved bridge pages to target documents.

Current graph construction:

```text
deduped_edge_count: 21451
source_doc_count: 2885 / 3366
target_doc_count: 2417 / 3366
gold_has_any_incoming_link_count: 1959 / 2441
gold_linked_from_baseline_sources_count: 1932 / 2441
```

Current best result:

```text
No-hyperlink baseline doc@4: 0.846, row@4: 0.758
PDF hyperlink w2.25 doc@4: 0.851, row@4: 0.770
MMR doc-diverse selection b0.10 doc@4: 0.846, row@4: 0.799
```

Interpretation:

- tuned `DOC_DOC_EDGE_WEIGHT=2.25` is the best balanced hyperlink setting observed so far
- larger hyperlink weights keep improving some row metrics but increasingly damage early doc rank and increase worsened qids
- final row selection and hyperlink edges solve different problems: hyperlink improves document propagation, while MMR/max-one-doc selection improves row diversity
- the next important test is the combination of tuned hyperlink `w2.25` with `FINAL_SELECTION_MODE=mmr_doc_diverse` and `FINAL_SELECTION_NEW_DOC_BONUS=0.10`

This is thesis-facing because the graph edges come from authored PDF/Wikipedia structure, not from labels. It is strongest on entity-chain and ImageListQ-style questions.

Latest M3DocVQA complete ablation notes:

- `shared_entity_title_topic` remains a no-op: `active_edge_qids=0`.
- `dense_sparse_agreement` and `fully_connected_topdocs` do not beat the tuned hyperlink setting.
- `docseed_*` rows hurt M3DocVQA doc@4 and should not be main settings.
- `semantic_similarity` was skipped only because the complete wrapper did not see `SPLADE_INDEX_PT`; the full index exists at `/mmfs1/scratch/jacks.local/aerfanshekooh/custom/outputs/m3docvqa_splade/m3docvqa_dev_splade.pt`.

### 7. News Query-Anchor Audit

Purpose: understand why query anchors help some News questions but also create many rank regressions.

Latest audit:

```text
News qids: 137
recovered: 2
improved_rank: 38
worsened_rank: 29
missing_in_both: 14
main limitations:
  competing_top_pages_match_as_many_or_more_anchors: 45
  no_query_anchors: 26
  gold_page_has_anchor_evidence_but_not_in_candidate_head: 19
```

Conclusion:

- pure anchor matching is too weak for topical/news redundancy
- stronger global anchor weights are likely to increase false positives
- the non-heuristic path is learned evidence reliability or a query-conditioned verifier
- use the audit as the motivation for adaptive graph evidence calibration

## Reporting Rule

For each new method, always report:

1. Overall page_hit@4 and doc_hit@4.
2. Recovered/lost/improved/worsened counts versus graph baseline.
3. Domain/type breakdown.
4. Whether the gold page existed in dense, sparse, either, or neither source pool.
5. Method activation counts from summary JSON.

This prevents us from claiming novelty without measurable behavior.
