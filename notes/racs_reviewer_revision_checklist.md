# RACS reviewer-request checklist for the advisor's first revision

As of 2026-09-18. This is an internal handoff, not a submitted response letter.
"Draft ready" means separate manual-insertion text exists, not that Overleaf
was edited. Nothing has been sent to the advisor or conference.

## Evidence ready for the first revision

| Reviewer request | Evidence / revision available | Remaining condition |
| --- | --- | --- |
| R1.1: feature selection and relation to prior work | Corrected 5/4/6/15 feature-family description; explicit rationale; distinction between established ranking/fusion ideas and the study's selected feature set | Author review and manual insertion; no claim of exhaustive feature search |
| R1.2: actual cost after GPP | Validated cached-input CPU benchmark: 196.2 ms/query, feature-stage breakdown, throughput and whole-process memory | Only partial coverage: end-to-end GPP-versus-CAPP latency and incremental memory are not measured |
| R1.3: competitive reranker under common pool and reader budget | Saved top-1000 BGE, monoT5 and LambdaMART retrieval reports confirmed; BGE job 15911612 completed with candidate/cohort/selected-page validation, 2441 questions, four pages each, 37.85 EM / 43.89 F1 | Manual table ready; alpha-selection provenance and matched total cost remain separate; no matched LambdaMART/monoT5 QA is reported |
| R1.4: lightweight architecture trade-offs | Scorer-capacity explanation; distinction between parameter count and feature cost; expanded ablation/QA results | Do not claim a demonstrated speed advantage over neural baselines or causal coefficient interpretation |
| R1.5: systematic feature ablation | Six variants verified, including removal of each of four families; fixed alpha 0.40; missing structure + content QA row supplied | Literal auxiliary-source filenames for historical source-bearing ablations are not established by matching main input paths |
| R1.6: sensitivity to reader budget | k=1,2,4,8 QA for GPP/CAPP on the same 2,441 qids; original k=4 reused | Historical GPU timings are descriptive, not controlled cross-hardware speed comparisons |
| R2: gold-page injection / learning the labeler | Matched 2,188-question, exactly-four-page pseudo-gold injection: F1 51.39 vs CAPP 44.54 and GPP 42.34 | Pseudo-gold, not human-gold; supports answer usefulness, does not eliminate labeler--feature coupling |
| R2: content-aware name and absent QA comparison | Explain that content is included but not dominant; ablation table gives structure + content 45.14 F1 versus full 45.69 | No statistical superiority claim; do not substitute all-question scores for matched-subset scores |

Controlling review source: the user's acceptance/review email at
`/Users/hoseinerfan/.codex/attachments/ca3c4f4d-1c7d-4e0a-b3b0-732c3a664e9b/pasted-text.txt`.
R1 has six numbered requests; R2 adds label-coupling and naming/comparison issues.

## Manual integration files

Start with `notes/racs_overleaf_complete_revision_packet.md`: it contains the
full ordered insertion/replacement packet, with alpha provenance explicitly
unverified. Do not paste both that packet and the duplicate fragments below.
The following files remain the supporting evidence and original draft guides.

1. `notes/racs_manual_revisions_runtime_budget_control.md`: runtime, budget,
   and fixed-four-page injection. Keep the main-result and matched-subset
   populations separate.
2. `notes/racs_manual_revisions_baselines_ablation.md`: expanded ablation table
   and interpretation, plus the validated BGE four-page QA comparison.
3. `notes/racs_manual_revisions_feature_rationale.md`: corrected feature table,
   conceptual citations, source-input clarification, trade-offs, Discussion,
   and consolidated Limitations. Its full Limitations replacement supersedes
   the first guide's piecemeal Limitations insertions.

## Must not disappear from the advisor handoff

- **Full-model provenance:** the verified evaluation replay uses no-hyperlink
  GPP as its base and three auxiliary GPP rankings, two with active hyperlink
  edges. This is distinct from the author's recollection of an entirely
  hyperlink-free final system. The no-source ablation is a separate model.
  Probe 15911725 also confirms an approximate MaxSim dense input with a
  224-token page budget for the legacy auxiliaries; the main base uses Exact
  MaxSim. The manual protocol now makes this distinction explicit.
- **Alpha selection:** fixed 0.40 is verified; held-out-training selection and
  subsequent final retraining are not established by the available final
  artifacts. The thesis explicitly documents a training-only 20% holdout,
  page@4 selection, and full-training refit (Chapter 4, lines 230–234), as does
  the paper. CPU-only inventory job 15911623 completed and its full report was
  inspected. Matching-input fixed-alpha 0.35/0.40/0.45 runs exist, but their
  tuning summaries are null. This supports the existence of trials, not the
  selection split/criterion. BGE's alpha-0.20 selection provenance is also open.
  The author answered "I am not sure" on September 18; the packet therefore
  does not present training-only tuning as established fact.
- **Scope of runtime evidence:** the CPU result is validated but is not a full
  pipeline overhead comparison. A statement of remaining scope is necessary;
  it does not by itself fully satisfy R1.2.
- **Baseline result:** accounting and `BGE_READER_RESULT` confirm the completed
  BGE reader run and its output checks. This adds one competitive baseline's
  matched-cohort/four-page QA, not statistical significance or evidence of
  superiority over every configuration. No matched LambdaMART/monoT5 reader
  scores are reported yet.
- **Final artifact:** compile the manually updated Overleaf source, inspect all
  tables/references, and check the applicable page limit. No compiled revised
  manuscript exists from these writing-only changes.

## Suggested sequence from here

Integrate the manual prose/ablation and now-complete BGE QA comparison, and
review the provenance questions with the advisor. The runtime audit JSON is
now inspected and the legacy dense core settings are confirmed. The prepared
16-question/four-graph replay is the next check before a faithful pipeline
benchmark; it is not the benchmark itself.
After manual insertion, send the advisor the first compiled revision with
this concise list of remaining decisions. Further training is not needed for
these writing changes or to use the completed BGE run. Any new controlled
end-to-end benchmark is a separate experiment, not something already done.
