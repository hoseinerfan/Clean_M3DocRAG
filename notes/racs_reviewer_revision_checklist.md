# RACS reviewer-request checklist for the advisor's first revision

As of 2026-09-17. This is an internal handoff, not a submitted response letter.
"Draft ready" means separate manual-insertion text exists, not that Overleaf
was edited. Nothing has been sent to the advisor or conference.

## What can be completed while BGE runs

| Reviewer request | Evidence / revision available | Remaining condition |
| --- | --- | --- |
| R1.1: feature selection and relation to prior work | Corrected 5/4/6/15 feature-family description; explicit rationale; distinction between established ranking/fusion ideas and the study's selected feature set | Author review and manual insertion; no claim of exhaustive feature search |
| R1.2: actual cost after GPP | Validated cached-input CPU benchmark: 196.2 ms/query, feature-stage breakdown, throughput and whole-process memory | Only partial coverage: end-to-end GPP-versus-CAPP latency and incremental memory are not measured |
| R1.3: competitive reranker under common pool and reader budget | Saved top-1000 BGE, monoT5 and LambdaMART retrieval reports confirmed; BGE four-page reader job 15911612 submitted | Await job success and candidate/cohort/selected-page checks, then add QA; alpha-selection provenance and matched total cost remain separate |
| R1.4: lightweight architecture trade-offs | Scorer-capacity explanation; distinction between parameter count and feature cost; expanded ablation/QA results | Do not claim a demonstrated speed advantage over neural baselines or causal coefficient interpretation |
| R1.5: systematic feature ablation | Six variants verified, including removal of each of four families; fixed alpha 0.40; missing structure + content QA row supplied | Literal auxiliary-source filenames for historical source-bearing ablations are not established by matching main input paths |
| R1.6: sensitivity to reader budget | k=1,2,4,8 QA for GPP/CAPP on the same 2,441 qids; original k=4 reused | Historical GPU timings are descriptive, not controlled cross-hardware speed comparisons |
| R2: gold-page injection / learning the labeler | Matched 2,188-question, exactly-four-page pseudo-gold injection: F1 51.39 vs CAPP 44.54 and GPP 42.34 | Pseudo-gold, not human-gold; supports answer usefulness, does not eliminate labeler--feature coupling |
| R2: content-aware name and absent QA comparison | Explain that content is included but not dominant; ablation table gives structure + content 45.14 F1 versus full 45.69 | No statistical superiority claim; do not substitute all-question scores for matched-subset scores |

Controlling review source: the user's acceptance/review email at
`/Users/hoseinerfan/.codex/attachments/ca3c4f4d-1c7d-4e0a-b3b0-732c3a664e9b/pasted-text.txt`.
R1 has six numbered requests; R2 adds label-coupling and naming/comparison issues.

## Manual integration files

1. `notes/racs_manual_revisions_runtime_budget_control.md`: runtime, budget,
   and fixed-four-page injection. Keep the main-result and matched-subset
   populations separate.
2. `notes/racs_manual_revisions_baselines_ablation.md`: expanded ablation table
   and interpretation; baseline QA awaits BGE job 15911612.
3. `notes/racs_manual_revisions_feature_rationale.md`: corrected feature table,
   conceptual citations, source-input clarification, trade-offs, Discussion,
   and consolidated Limitations. Its full Limitations replacement supersedes
   the first guide's piecemeal Limitations insertions.

## Must not disappear from the advisor handoff

- **Full-model provenance:** the verified evaluation replay uses no-hyperlink
  GPP as its base and three auxiliary GPP rankings, two with active hyperlink
  edges. This is distinct from the author's recollection of an entirely
  hyperlink-free final system. The no-source ablation is a separate model.
- **Alpha selection:** fixed 0.40 is verified; held-out-training selection and
  subsequent final retraining are not established by the available final
  artifacts. The thesis explicitly documents a training-only 20% holdout,
  page@4 selection, and full-training refit (Chapter 4, lines 230–234), as does
  the paper. A CPU-only inventory is prepared to locate any corresponding
  earlier tuning record; this distinguishes documented protocol from execution
  evidence. BGE's alpha-0.20 selection provenance is also open.
- **Scope of runtime evidence:** the CPU result is validated but is not a full
  pipeline overhead comparison. A statement of remaining scope is necessary;
  it does not by itself fully satisfy R1.2.
- **Baseline result:** do not claim the new BGE reader run is successful until
  accounting shows completion and the job prints `BGE_READER_RESULT` after
  its output checks. No matched LambdaMART/monoT5 reader scores are reported yet.
- **Final artifact:** compile the manually updated Overleaf source, inspect all
  tables/references, and check the applicable page limit. No compiled revised
  manuscript exists from these writing-only changes.

## Suggested sequence from here

Finish manual prose/ablation integration and review the two author-confirmation
questions while job 15911612 proceeds. After its validated EM/F1 arrives, finish
the baseline comparison. Then send the advisor the first compiled revision
with this concise list of remaining decisions. Further training is not needed
for these writing changes or the pending frozen-reader run. Any new controlled
end-to-end benchmark would be a separate experiment, not something already done.
