# RACS: closing the remaining revision items

Updated 2026-09-18 after the author requested all remaining items.
This plan does not change Overleaf or launch remote jobs automatically.

## 1. BGE reader result

Job 15911612 is complete. On September 18 the author provided `COMPLETED`,
exit `0:0`, elapsed `02:26:55`, node `gpu009`, and the postflight result:
2441 questions, four pages each, EM 37.85333879557559 / F1 43.8947152806227.
The stdout marker agrees with
`output/racs_bge_reader_top4_15911612/validated_result.json`.
The baseline guide now includes 37.85 EM / 43.89 F1 in its manual table.
Do not submit a duplicate. This is evidence from the completed validated HPC
run, not a local rescore of its full predictions. Tuning provenance and matched
end-to-end runtime remain separate; this result does not resolve either.

## 2. Alpha selection: written protocol located

The author asked whether the thesis/paper already explains the procedure. Yes:

- Thesis `sdsu_dissertation_template/section3.tex`, lines 314–318: training-only
  20% holdout, maximizing page@4, then full-training refit.
- Thesis `sdsu_dissertation_template/section4.tex`, lines 230–234: same procedure.
- Thesis `sdsu_dissertation_template/section5.tex`, lines 833–838: repeats it.
- September 15 manuscript `main.tex`, Method and Training paragraphs: held-out
  training selection for page@4 followed by final retraining.

This confirms the **documented protocol**, not an independently recovered
execution trace. The final full-model artifact records fixed alpha 0.40 and no
automatic tuning in that run. This does not disprove an earlier selection run.
The provenance note also records a historical recommendation for separate fixed
alpha trials; a recommended command is not evidence of how the final choice
was made. These distinctions should not be collapsed in either direction.

Next: inventory relevant small model/summary files for prior training-only
tuning records. A candidate must match the label version, training/base inputs,
feature set and selected alpha; a different model's 0.40 choice is not proof.
Matching fit/tune counts alone is also insufficient. If no record is recovered,
ask the author/advisor to confirm the written protocol or qualify it. Do not
silently invent a development-tuned or train-tuned history. No retraining is
authorized by this inventory, and a new tuning run cannot establish history.

## 3. Full-model source description

The manual feature-rationale guide includes the corrected evaluation protocol:
no-hyperlink Exact MaxSim GPP base, plus three legacy auxiliary rankings,
including two hyperlink-enabled variants. The full model and the no-source
ablation remain distinct. This text can be reviewed with the advisor now.
The runtime comparison must charge the full method for whichever auxiliary
generation it needs, rather than assuming saved rankings are free.

## 4. Controlled runtime / memory comparison

The existing 196.2 ms/query result is a valid cached-input CAPP measurement,
not a full query-to-answer benchmark. Do not sum historical timers recorded
on different nodes into a purported end-to-end comparison.

### Prerequisite audit (job 15911623 completed)

`examples/sbatch_racs_runtime_prerequisites.sh` requests one CPU and 64 GB on
`compute`, with a 30-minute cap. It loads the four graph artifacts one at a
time, records their saved configuration, checks qid coverage and referenced
file availability, and inventories the upstream dense/sparse summaries. It
also inventories relevant full-feature CAPP tuning records and any completed
BGE result. Only a new `output/racs_runtime_audit_JOBID/` report is written.

This is a configuration check, **not** a new benchmark or model run. Saved
timing fields remain labeled as unverified in scope. Missing recorded paths
may be optional/unused and must be interpreted with the saved graph mode.
Large metadata files skipped by the size limit are reported as unread, not
absent. The current graph wrapper defaults are not treated as historical truth.

User-provided audit output now confirms job 15911623 completed with exit 0:0
in 20 seconds on node011. All four graph artifacts exist and their qid sets
match the 2,441-question gold file; no missing direct recorded dependency paths
were reported. Evidence:
`/Users/hoseinerfan/.codex/attachments/edc16373-5d03-4af8-b94c-e65a5b8f14c6/pasted-text.txt`.

Important configuration distinctions from that output:

- The main base uses `mmqa_dev_exact_maxsim_nprobe4_ret1000.prediction.json`
  and final selection `score`.
- All three auxiliary rankings instead use
  `mmqa_dev_plain_top224_nprobe4_effdiag_all.prediction.json` and final
  selection `mmr_doc_diverse`. Their graph input/output limits are still
  1,000 pages; the filename alone does not establish a different graph limit.
- The shared sparse artifact's saved summary identifies
  `naver/splade-cocondenser-ensembledistil`, top 1,000 pages and 32 query terms.
  This is not the earlier recollection of SPLADE v3. Preserve the recorded
  checkpoint identity unless more direct evidence establishes stale metadata;
  do not silently substitute a v3 artifact in a reproduction or benchmark.
- The legacy dense prediction file exists, but its expected same-stem summary
  is absent. This is a provenance gap, not a missing ranking file.

These checks establish available artifacts and recorded configuration, not
recomputed graph equivalence, transitive model/index availability, or a runtime
result. The full report on HPC is
`output/racs_runtime_audit_15911623/prerequisites.json`. Transfer that report
for local inspection of complete graph settings, varying metadata keys and
tuning records; no duplicate audit is needed. The pasted filtered output
contains no BGE completion marker or alpha-tuning records.

### Benchmark design after prerequisite verification

1. Freeze the exact shared and method-specific inputs. GPP must reproduce the
   paper's base rankings. Full CAPP must reproduce its saved candidate sets
   and ranking order using the verified three auxiliary sources. Rebuild a
   small fixed-qid sample first, failing on differences rather than choosing
   a nearby configuration with favorable scores.
2. Compare GPP+reader and full-CAPP+reader on the same hardware, qids, rendered
   pages, reader checkpoint, precision, prompt and four-page budget. Record
   GPU model/driver, CPU allocation, software/code hashes, cache policy and
   resident models. Select any benchmark subset before looking at outcomes;
   report its size and do not present it as a full-development-set result.
3. Define the start/end boundaries explicitly. A genuine online query-to-answer
   benchmark includes query encoding/search/exact scoring, required graph
   ranking(s), CAPP features/scoring where applicable, image preparation and
   answer generation. Offline corpus indexing/page extraction and one-time
   initialization should be reported separately. If a stage cannot be rebuilt,
   label the result as a partial pipeline benchmark rather than end-to-end.
4. Include auxiliary branches unique to full CAPP, and charge any additional
   retrieval used by those branches. Shared work can be reused within a query
   but must not be double-counted or omitted differently across conditions.
   Do not assume the legacy auxiliary ranks use the same dense retrieval as
   the Exact MaxSim base just because both are GPP outputs.
5. Warm up explicitly, then repeat measured passes with balanced method order.
   Synchronize GPU operations at timing boundaries. Report per-query mean,
   median/p95 and serial throughput, plus initialization/preparation costs.
   Keep validation and result serialization outside the declared query timer.
6. Measure the conditions in separate fresh worker processes on the same
   allocation. Record CPU peak RSS and GPU peak allocated/reserved memory
   separately. Compare matched condition-level peaks; do not subtract two
   monotonically increasing high-water marks within one reused process or
   describe memory for validation-reference JSON as deployed scorer memory.
7. Publish runtime differences only after ranking/cohort checks pass. Explain
   any reader answer variation; do not replace the paper's established QA
   results merely because a timing run generates new answers.

The actual paired benchmark launcher is not yet implemented: reconstructing
the upstream branches requires inspecting the complete audit report and
resolving the legacy dense configuration. This is a concrete outstanding
step, not evidence that the runtime reviewer request is complete.

Local verification: five audit unit tests cover question-ID conflicts, graph
configuration variation, absent/oversized metadata, output isolation and
tuning-record filtering. The Slurm launcher passes Bash syntax checking.

## 5. Manual manuscript integration

Use the three manual guides in this order: runtime/budget/control;
baselines/ablation; feature rationale. The consolidated Limitations body in
the third guide replaces the earlier piecemeal limitations edits. Preserve
the original main-table cohort and do not paste matched-subset QA into it.
The baseline guide now includes a combined retrieval/QA LaTeX table and BGE
model-card entry. BGE's validated four-page QA is 37.85 EM / 43.89 F1; QA for
LambdaMART and monoT5 remains unreported. Replace the earlier retrieval-only
draft rather than inserting both versions of the same subsection/table.

Original Overleaf source/ZIP remains unchanged. The author requested manual
integration; no rewritten or compiled revised manuscript is claimed. After
manual insertion, compile and inspect table widths, numbering, citations,
float placement, and page limit, then share the first draft with the advisor.
If the author wants direct work on a separate manuscript copy, that needs an
explicit change to the earlier manual-only instruction.

## Next handoff: transfer the completed audit report

No second audit submission is needed. In a new local Mac terminal (not inside
the HPC SSH session), copy the existing report:

~~~bash
scp -o 'User=aerfanshekooh@jacks.local' \
  innovator.sdstate.edu:/mmfs1/scratch/jacks.local/aerfanshekooh/custom/Clean_M3DocRAG/output/racs_runtime_audit_15911623/prerequisites.json \
  /Users/hoseinerfan/Desktop/Clean_M3DocRAG/outputs/racs_runtime_prerequisites_15911623.json
~~~

This transfers only the audit JSON, not the large predictions. The destination
was absent at the local check on September 18. Once available, inspect the
remaining configuration/tuning records directly. BGE job 15911612's completion
and validation result are now recorded separately above; do not rerun the
audit just to refresh its optional BGE snapshot. Writing-only notes updates
do not require any GPU job to be rerun.
