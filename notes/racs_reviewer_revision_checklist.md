# RACS reviewer-request completion checklist

As of 2026-09-18. This is an internal handoff, not a submitted response letter.
The author explicitly changed the target to **all reviewer requests completed**,
not an advisor-first-draft stopping point. Available manual text is not the same
as an integrated, checked paper. Nothing has been sent to the advisor/conference.

## Evidence available and closure conditions

| Reviewer request | Evidence / revision available | Remaining condition |
| --- | --- | --- |
| R1.1: feature selection and relation to prior work | Corrected 5/4/6/15 feature-family description; explicit rationale; distinction between established ranking/fusion ideas and the study's selected feature set | Author review and manual insertion; no claim of exhaustive feature search |
| R1.2: actual cost after GPP | Cached-input CPU CAPP: 196.2 ms/query. Matched graph-to-answer job 15911732: GPP 3.942 s / CAPP 4.703 s; raw-record arithmetic audit also passed | **Open experiment:** online-runtime implementation now includes dense/SPLADE retrieval and both scoring routes, but awaits HPC preflight/measurement. Whole-worker RSS is not isolated incremental scorer memory |
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

## Must remain explicit in the final revision

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
- **Scope of runtime evidence:** the CPU-stage result and now-completed matched
  graph-to-answer comparison are validated within their declared boundaries.
  The latter reports +0.761 s (+19.3%) for CAPP, but starts at cached dense/sparse
  rankings, including the distinct legacy auxiliary input. Neither establishes
  full online pipeline overhead or isolated scorer memory; R1.2 is not fully
  closed by relabeling these partial measurements.
- **Baseline result:** accounting and `BGE_READER_RESULT` confirm the completed
  BGE reader run and its output checks. This adds one competitive baseline's
  matched-cohort/four-page QA, not statistical significance or evidence of
  superiority over every configuration. No matched LambdaMART/monoT5 reader
  scores are reported yet.
- **Final artifact:** compile the manually updated Overleaf source, inspect all
  tables/references, and check the applicable page limit. No compiled revised
  manuscript exists from these writing-only changes.

## Completion sequence from here

Integrate the manual prose/ablation and now-complete BGE QA comparison, and
review the provenance questions with the advisor. The runtime audit JSON is
now inspected and the legacy dense core settings are confirmed. Job 15911730
passed the 16-question/four-graph replay: every full order and score comparison
matched within its declared tolerance. This is not the benchmark itself.
Job 15911732 subsequently completed graph-to-answer timing on 128 fixed questions,
including CAPP's three auxiliary graphs and the four-page reader on one A100.
Its supplied summaries reconcile and its reported status confirms validation.
The new result is ready for its declared scoped use; do not repeat that job.
It explicitly excludes dense/SPLADE retrieval and scoring, so it does not close
the online end-to-end runtime request. See
`notes/racs_graph_reader_runtime_protocol.md` for results and evidence boundaries.
The author also supplied `INDEPENDENT_RUNTIME_CHECK_PASSED` from the raw worker
records. That is a successful aggregation audit, not a new online timing run.

1. Run `examples/sbatch_racs_online_runtime.sh`. The new code performs an upstream
   equivalence preflight and then paired online timing with fail-closed checks.
   See `notes/racs_online_runtime_protocol.md`. No online numbers exist yet.
2. Audit the resulting raw timings and checks. Add only a validated online
   result to the manual packet and finalize the runtime/architecture discussion.
3. Apply the prepared feature-rationale, stronger-baseline, family-ablation,
   budget and injection revisions manually in Overleaf, preserving the explicit
   source/tuning/label limitations. BGE satisfies R1.3's request for one or more
   competitive rerankers; missing QA for every other baseline is not a mandatory
   new experiment.
4. Finalize the point-by-point response against the actual revised paper;
   compile, inspect tables/figures/citations/page limits and check every claimed
   change is present. Obtain coauthor review before final submission.

No retraining is required by these remaining experiments. If online replay
fails, diagnose that exact mismatch rather than changing the paper configuration
or treating an unvalidated timer as evidence. Unknown historical alpha selection
must remain honestly qualified; a new tuning run would be a new experiment,
not recovered history. The manuscript/ZIP remains untouched by this workflow.
